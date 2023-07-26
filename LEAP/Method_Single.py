import json
import sys
import torch
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from www.dataset.ann import att_to_num_classes
from sklearn.metrics import accuracy_score, f1_score
from www.dataset.ann import att_to_idx, att_to_num_classes, att_types,att_default_values


def compute_metrics(preds, labels, metrics):
    metr = {}
    for m, m_name in metrics:
        if m_name in ['accuracy', 'confusion_matrix']:
            metr[m_name] = m(
                labels, preds
            )  # Assume each metric m will be a function of (y_true, y_pred)
        else:
            metr[m_name] = m(labels, preds, average='macro')
    return metr

def getLengthMask(input_ids,input_lengths,input_entities):
    batch_size, num_stories, num_entities, num_sents, seq_length = input_ids.shape
    assert num_stories == 2

    input_lengths = input_lengths.view(-1)
    input_entities = input_entities.view(-1)

    length_mask = torch.ones(
        (batch_size * num_stories * num_entities, num_sents),
        requires_grad=False).to(input_lengths.device)
    for i in range(num_sents):
        length_mask[
            input_lengths <= i,
            i] = 0  # Use input lengths to zero out state and conflict preds wherever there isn't a sentence
    length_mask = length_mask.view(batch_size * num_stories, num_entities,
                                   num_sents)
    for i in range(num_entities):
        length_mask[
            input_entities <= i,
            i, :] = 0  # Use input entity counts to zero out state and conflict preds wherever there isn't an entity
    length_mask = length_mask.view(batch_size * num_stories * num_entities,
                                   num_sents)
    return length_mask

def update_result(prep_result, effe_result, original_input, input_lengths,
                  input_entities,num_attributes):
    batch_size, num_stories, num_entities, num_sents, seq_length = original_input.shape
    assert num_stories == 2

    input_lengths = input_lengths.view(-1)
    input_entities = input_entities.view(-1)

    length_mask = torch.ones(
        (batch_size * num_stories * num_entities, num_sents),
        requires_grad=False).to(input_lengths.device)
    for i in range(num_sents):
        length_mask[
            input_lengths <= i,
            i] = 0  # Use input lengths to zero out state and conflict preds wherever there isn't a sentence
    length_mask = length_mask.view(batch_size * num_stories, num_entities,
                                   num_sents)
    for i in range(num_entities):
        length_mask[
            input_entities <= i,
            i, :] = 0  # Use input entity counts to zero out state and conflict preds wherever there isn't an entity
    length_mask = length_mask.view(batch_size * num_stories * num_entities,
                                   num_sents)

    prep_result *= length_mask.view(-1).repeat(num_attributes, 1).t()
    assert length_mask.view(-1).shape[0] == prep_result.shape[0]

    effe_result *= length_mask.view(-1).repeat(num_attributes, 1).t() # Mask out any nonexistent entities or sentences
    assert length_mask.view(-1).shape[0] == effe_result.shape[0] 
    
    return prep_result, effe_result
def trainModel_config(batch,max_story_length,maxStoryLength,num_attributes,device,tslm,tslm_optimizer,story_label,grad_accmu,loss_percent):
    grad_count=1
    # Process in one batch
    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device)  #.to(torch.int64).to('cpu')
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0]
    
    # result of the whole story
    prep_result=torch.tensor([]).to(device)
    effe_result=torch.tensor([]).to(device)
    conflict_result=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}

    modify_input_ids = batch[0].view(-1, max_story_length, maxStoryLength).long().to(device)  # 28* 5 * 168
    modify_input_mask = batch[3].view(-1, max_story_length, maxStoryLength).to(device)        # 28* 5 * 168
    modify_preconditions = batch[5].view(-1, max_story_length,num_attributes).long().to(device)  # 28* 5 * 20
    modify_effects = batch[6].view(-1, max_story_length,num_attributes).long().to(device)      # 28* 5 * 20
    modify_timestep_type_ids=batch[9].view(-1, max_story_length, maxStoryLength).long().to(device)  # 28* 5 * 168
    modify_conflict = batch[7].view(-1,max_story_length).long().to(device)            # 28* 5
    if story_label:
        modify_labels=torch.tensor(int(len(modify_input_ids)/2)*[labels]+int(len(modify_input_ids)/2)*[1-labels]).long().to(device)
    else:
        modify_labels=all_entity_labels.view(-1)
    for entity_idx in range(len(modify_input_ids)):
        entity_input = modify_input_ids[entity_idx]
        entity_mask = modify_input_mask[entity_idx] 
        entity_timestep = modify_timestep_type_ids[entity_idx]
        entity_preconditions = modify_preconditions[entity_idx]
        entity_effects = modify_effects[entity_idx]
        entity_conflict= modify_conflict[entity_idx]
        entity_label= modify_labels[entity_idx].view(-1)
        
        #forward
        if int(active_entity_list[entity_idx].item()) == 0:
            entity_out={'out_preconditions': torch.zeros(modify_preconditions[0].shape).to(device),
             'out_effects' : torch.zeros(modify_preconditions[0].shape).to(device),
             'out_conflict': torch.zeros(int(len(modify_conflict[0])*(len(modify_conflict[0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
        else:
            entity_out=tslm(input_ids=entity_input,attention_mask=entity_mask,timestep_type_ids=entity_timestep\
                 ,token_type_ids=segment_ids,prec_label=entity_preconditions,effect_label=entity_effects,\
                conflict=entity_conflict,label=entity_label)

        #backward
            entity_loss=loss_percent[0]*entity_out['loss_preconditions']/20+loss_percent[1]*entity_out['loss_effects']/20\
            +loss_percent[2]*entity_out['loss_conflict']+loss_percent[3]*entity_out['loss_story']
            entity_loss=entity_loss/grad_accmu
            entity_loss.backward()
            if grad_count%grad_accmu==0:
                tslm_optimizer.step()
                tslm_optimizer.zero_grad()
            grad_count+=1

        
    
        prep_result = torch.cat((prep_result,entity_out['out_preconditions']),dim=0)
        effe_result = torch.cat((effe_result,entity_out['out_effects']),dim=0)
        conflict_result = torch.cat((conflict_result,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']

    # Summarize result
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/torch.sum(input_entities).item()
    final_out['total_loss']=total_entity_loss_total/torch.sum(input_entities).item()
    
    return final_out
def trainModel(batch,max_story_length,maxStoryLength,num_attributes,device,tslm,tslm_optimizer,story_label,grad_accmu):
    grad_count=1
    # Process in one batch
    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device)  #.to(torch.int64).to('cpu')
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0]
    
    # result of the whole story
    prep_result=torch.tensor([]).to(device)
    effe_result=torch.tensor([]).to(device)
    conflict_result=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}

    modify_input_ids = batch[0].view(-1, max_story_length, maxStoryLength).long().to(device)  # 28* 5 * 168
    modify_input_mask = batch[3].view(-1, max_story_length, maxStoryLength).to(device)        # 28* 5 * 168
    modify_preconditions = batch[5].view(-1, max_story_length,num_attributes).long().to(device)  # 28* 5 * 20
    modify_effects = batch[6].view(-1, max_story_length,num_attributes).long().to(device)      # 28* 5 * 20
    modify_timestep_type_ids=batch[9].view(-1, max_story_length, maxStoryLength).long().to(device)  # 28* 5 * 168
    modify_conflict = batch[7].view(-1,max_story_length).long().to(device)            # 28* 5
    if story_label:
        modify_labels=torch.tensor(int(len(modify_input_ids)/2)*[labels]+int(len(modify_input_ids)/2)*[1-labels]).long().to(device)
    else:
        modify_labels=all_entity_labels.view(-1)
    for entity_idx in range(len(modify_input_ids)):
        entity_input = modify_input_ids[entity_idx]
        entity_mask = modify_input_mask[entity_idx] 
        entity_timestep = modify_timestep_type_ids[entity_idx]
        entity_preconditions = modify_preconditions[entity_idx]
        entity_effects = modify_effects[entity_idx]
        entity_conflict= modify_conflict[entity_idx]
        entity_label= modify_labels[entity_idx].view(-1)
        
        #forward
        if int(active_entity_list[entity_idx].item()) == 0:
            entity_out={'out_preconditions': torch.zeros(modify_preconditions[0].shape).to(device),
             'out_effects' : torch.zeros(modify_preconditions[0].shape).to(device),
             'out_conflict': torch.zeros(int(len(modify_conflict[0])*(len(modify_conflict[0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
        else:
            entity_out=tslm(input_ids=entity_input,attention_mask=entity_mask,timestep_type_ids=entity_timestep\
                 ,token_type_ids=segment_ids,prec_label=entity_preconditions,effect_label=entity_effects,\
                conflict=entity_conflict,label=entity_label)

        #backward
            entity_loss=entity_out['total_loss']/grad_accmu
            entity_loss.backward()
            if grad_count%grad_accmu==0:
                tslm_optimizer.step()
                tslm_optimizer.zero_grad()
            grad_count+=1

        
    
        prep_result = torch.cat((prep_result,entity_out['out_preconditions']),dim=0)
        effe_result = torch.cat((effe_result,entity_out['out_effects']),dim=0)
        conflict_result = torch.cat((conflict_result,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']

    # Summarize result
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/torch.sum(input_entities).item()
    final_out['total_loss']=total_entity_loss_total/torch.sum(input_entities).item()
    
    return final_out


def evalModel(batch,max_story_length,maxStoryLength,num_attributes,device,tslm,tslm_optimizer,story_label):
    # Process in one batch

    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device)  #.to(torch.int64).to('cpu')
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0]

    # result of the whole story
    prep_result=torch.tensor([]).to(device)
    effe_result=torch.tensor([]).to(device)
    conflict_result=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}

    modify_input_ids = batch[0].view(-1, max_story_length, maxStoryLength).long().to(device)
    modify_input_mask = batch[3].view(-1, max_story_length, maxStoryLength).to(device)
    modify_preconditions = batch[5].view(-1, max_story_length,num_attributes).long().to(device)
    modify_effects = batch[6].view(-1, max_story_length,num_attributes).long().to(device)
    modify_timestep_type_ids=batch[9].view(-1, max_story_length, maxStoryLength).long().to(device)
    modify_conflict = batch[7].view(-1,max_story_length).long().to(device)
    if story_label:
        modify_labels=torch.tensor(int(len(modify_input_ids)/2)*[labels]+int(len(modify_input_ids)/2)*[1-labels]).long().to(device)
    else:
        modify_labels=all_entity_labels.view(-1)
    for entity_idx in range(len(modify_input_ids)):
        entity_input = modify_input_ids[entity_idx]
        entity_mask = modify_input_mask[entity_idx] 
        entity_timestep = modify_timestep_type_ids[entity_idx]
        entity_preconditions = modify_preconditions[entity_idx]
        entity_effects = modify_effects[entity_idx]
        entity_conflict= modify_conflict[entity_idx]
        entity_label= modify_labels[entity_idx].view(-1)
        #forward
        if int(active_entity_list[entity_idx].item()) == 0:
            entity_out={'out_preconditions': torch.zeros(modify_preconditions[0].shape).to(device),
                 'out_effects' : torch.zeros(modify_preconditions[0].shape).to(device),
                 'out_conflict': torch.zeros(int(len(modify_conflict[0])*(len(modify_conflict[0])-1)/2)).to(device),
                 'out_story' : torch.zeros([2]).to(device),
                 'loss_preconditions': 0,
                 'loss_effects': 0, 
                 'loss_conflict' : 0,
                 'loss_story' : 0,
                 'total_loss' : 0}
        else:
            with torch.no_grad():
                entity_out=tslm(input_ids=entity_input,attention_mask=entity_mask,timestep_type_ids=entity_timestep\
                     ,token_type_ids=segment_ids,prec_label=entity_preconditions,effect_label=entity_effects,\
                    conflict=entity_conflict,label=entity_label)
        
        

        prep_result = torch.cat((prep_result,entity_out['out_preconditions']),dim=0)
        effe_result = torch.cat((effe_result,entity_out['out_effects']),dim=0)
        conflict_result = torch.cat((conflict_result,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']

    # Summarize result
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/torch.sum(input_entities).item()
    final_out['total_loss']=total_entity_loss_total/torch.sum(input_entities).item()
    
    return final_out

def evaluation(max_story_length,maxStoryLength,num_attributes,tslm,tslm_optimizer,dev_dataloader,device,story_label,conflict_pred):
    # EVAL
    tslm.zero_grad()
    tslm.eval()
    if story_label:
        print("Use story label as the training label")
    else:
        print("Use entity label as the training label")

    if conflict_pred:
        print("Use conflict result to predict the story")
    else:
        if story_label:
            print("Use story result itself to predict the story")
        else:
            print("Use entity result itself to predict the story")


    all_pred_prec = None
    all_prec = None

    all_pred_eff = None
    all_eff = None

    all_pred_conflicts = None
    all_pred_conflicts_out=None
    all_conflicts = None

    all_pred_stories = None
    all_stories = None  
    all_pred_result=None
    
    for batch in tqdm(dev_dataloader):

    
        input_ids = batch[0].long().to(device)
        input_entities = batch[2].to(device)
        preconditions = batch[5].long().to(device)
        effects = batch[6].long().to(device)
        conflicts = batch[7].long().to(device)
        labels = batch[8].long().to(device)

        final_out=evalModel(batch,max_story_length,maxStoryLength,num_attributes,device,tslm,tslm_optimizer,story_label)

        #precondition

        label_ids = preconditions.view(-1, preconditions.shape[-1]).to('cpu').numpy()
        if all_prec is None:
            all_prec = label_ids
        else:
            all_prec = np.concatenate((all_prec, label_ids), axis=0)

        preds = final_out['out_preconditions'].detach().cpu().numpy()
        if all_pred_prec is None:
            all_pred_prec = preds
        else:
            all_pred_prec = np.concatenate((all_pred_prec, preds), axis=0)

        #effect

        label_ids = effects.view(-1, effects.shape[-1]).to('cpu').numpy()
        if all_eff is None:
            all_eff = label_ids
        else:
            all_eff = np.concatenate((all_eff, label_ids), axis=0)

        preds = final_out['out_effects'].detach().cpu().numpy()
        if all_pred_eff is None:
            all_pred_eff = preds
        else:
            all_pred_eff = np.concatenate((all_pred_eff, preds), axis=0)

        # conflict
        label_ids = conflicts.to('cpu').numpy()
        if all_conflicts is None:
            all_conflicts = label_ids
        else:
            all_conflicts = np.concatenate((all_conflicts, label_ids), axis=0)

        label_ids = final_out['out_conflicts'].view(1,conflicts.shape[1]*conflicts.shape[2],-1).detach().cpu().numpy()
        if all_pred_conflicts_out is None:
            all_pred_conflicts_out = label_ids
        else:
            all_pred_conflicts_out = np.concatenate((all_pred_conflicts_out, label_ids), axis=0)    



        preds=[]
        for index,item in enumerate(final_out['out_conflicts']):
            pair_index=torch.argmax(item)
            possiblity=item[pair_index]
            temp=[0]*max_story_length
            if possiblity>0.1:
                count=0
                flag_i=0
                flag_j=0
                for i in range(max_story_length):
                    for j in range(i+1,max_story_length):
                        if count==pair_index:
                            flag_i=i
                            flag_j=j
                        count+=1
                temp[flag_i]=1
                temp[flag_j]=1
            preds.append(temp)
        if all_pred_conflicts is None:
            all_pred_conflicts = preds
        else:
            all_pred_conflicts = np.concatenate((all_pred_conflicts, preds), axis=0)        

        # Stroy
        label_ids = labels.to('cpu').numpy()
        if all_stories is None:
            all_stories = label_ids
        else:
            all_stories = np.concatenate((all_stories, label_ids), axis=0)
            
        if conflict_pred:
            story_pred=torch.sum(final_out['out_story'].view(2,-1,2),dim=(1))
            conflict_result=final_out['out_conflicts'].view(2,-1,final_out['out_conflicts'].shape[-1])
            story_out= -torch.sum(conflict_result, dim=(1,2))
            story_out[0]/=input_entities[0][0]  # average over entity number
            story_out[1]/=input_entities[0][1]
            story_result=story_pred.view(1,2,2).cpu().numpy()
            preds=torch.argmax(story_out).view(-1,1).to('cpu').numpy()
        else:
            story_pred=torch.sum(final_out['out_story'].view(2,-1,2),dim=(1))
            story_pred[0]/=input_entities[0][0]  # average over entity number
            story_pred[1]/=input_entities[0][1]
            story_result=story_pred.view(1,2,2).cpu().numpy()
            if story_label:   # 0 represent plausbile  story tag
                if story_pred[0][0]>story_pred[1][0]:
                    preds=0
                else:
                    preds=1
                preds=np.array([[preds]])
            else:      # 0 means the entity in this story is implausible
                if story_pred[0][0]>story_pred[1][0]:
                    preds=1
                else:
                    preds=0
                preds=np.array([[preds]])
        if all_pred_result is None:
            all_pred_result = story_result
        else:
            all_pred_result = np.concatenate((all_pred_result, story_result), axis=0)
        if all_pred_stories is None:
            all_pred_stories = preds
        else:
            all_pred_stories = np.concatenate((all_pred_stories, preds), axis=0)
            
    metrics = [(accuracy_score, 'accuracy'),(f1_score, 'f1')]

    metr_prec = compute_metrics(all_pred_prec.flatten(), all_prec.flatten(), metrics)
    for i in range(num_attributes):
        metr_i = compute_metrics(all_pred_prec[:, i], all_prec[:, i], metrics)
        for k in metr_i:
            metr_prec['%s_%s' % (str(k), str(i))] = metr_i[k]

    metr_eff = compute_metrics(all_pred_eff.flatten(), all_eff.flatten(), metrics)
    for i in range(num_attributes):
        metr_i = compute_metrics(all_pred_eff[:, i], all_eff[:, i], metrics)
        for k in metr_i:
            metr_eff['%s_%s' % (str(k), str(i))] = metr_i[k]

    metr_conflicts = compute_metrics(all_pred_conflicts.flatten(), all_conflicts.flatten(), metrics)

    metr_stories = compute_metrics(all_pred_stories.flatten(), all_stories.flatten(), metrics)

    verifiability,consistency, explanations = verifiable_reasoning(max_story_length,all_stories,all_pred_stories,all_conflicts,all_pred_conflicts_out,all_prec,all_pred_prec,all_eff,all_pred_eff)
    
    for index,item in enumerate(explanations):
        item['story_result']=all_pred_result[index].tolist()
        for key in ['preconditions_label','preconditions_pred','effects_label','effects_pred']:
            item[key]=item[key].tolist()
    return metr_prec,metr_eff,metr_conflicts,metr_stories,verifiability,consistency, explanations

def verifiable_reasoning(max_story_length,all_stories,all_pred_stories,all_conflicts,all_pred_conflicts_out,all_prec,all_pred_prec,all_eff,all_pred_eff):
    atts = list(att_default_values.keys())
    verifiable = 0
    consistent_preds =0
    total = 0
    explanations = []
    for index, ex in enumerate(all_stories):
        l_story = all_stories[index]
        p_story = all_pred_stories[index]
        l_conflict = np.sum(all_conflicts, axis=(1,2))[index]
        l_conflict = np.nonzero(l_conflict)[0]
        p_conflict = all_pred_conflicts_out[index]
        p_index=np.argmax(np.mean(p_conflict,axis=0))
        temp=[0]*max_story_length
        count=0
        flag_i=0
        flag_j=0
        for i in range(max_story_length):
            for j in range(i+1,max_story_length):
                if count==p_index:
                    flag_i=i
                    flag_j=j
                count+=1
        temp[flag_i]=1
        temp[flag_j]=1
        p_conflict=temp
        p_conflict = np.nonzero(p_conflict)[0]
        l_prec = all_prec.reshape(list(all_conflicts.shape[:4]) + [all_prec.shape[-1]])[index,1-l_story] # (num entities, num sentences, num attributes)
        p_prec = all_pred_prec.reshape(list(all_conflicts.shape[:4]) + [all_prec.shape[-1]])[index,1-l_story] # (num entities, num sentences, num 
        l_eff = all_eff.reshape(list(all_conflicts.shape[:4]) + [all_eff.shape[-1]])[index,1-l_story] # (num entities, num sentences, num attributes)
        p_eff = all_pred_eff.reshape(list(all_conflicts.shape[:4]) + [all_eff.shape[-1]])[index,1-l_story] # (num entities, num sentences, num attributes)
        explanation = {'story_label': int(l_story),
               'story_pred': int(p_story),
               'conflict_label': [int(c) for c in l_conflict],
               'conflict_pred': [int(c) for c in p_conflict],
               'preconditions_label': l_prec,
               'preconditions_pred': p_prec,
               'effects_label': l_eff,
               'effects_pred': p_eff,
               'valid_explanation': False}
        
        if l_story == p_story:
            if l_conflict[0] == p_conflict[0] and l_conflict[1] == p_conflict[1]:
                consistent=True
                states_verifiable = True
                found_states = False

                # Check that effect of first conflict sentence has states which are correct
                for sl, sp in [(l_eff, p_eff)]:  # Check preconditions and effects
                    for sl_e, sp_e in zip(sl, sp):  # Check all entities
                        for si in [l_conflict[0]]:  # Check conflicting sentences
                            sl_es = sl_e[si]
                            sp_es = sp_e[si]
                            for j, p in enumerate(
                                    sp_es
                            ):  # Check all attributes where there's a nontrivial prediction
                                if p != att_default_values[atts[
                                        j]] and p > 0:  # NOTE: p > 0 is required to avoid counting any padding predictions.
                                    found_states = True
                                    if p != sl_es[j]:
                                        states_verifiable = False

                # Check that precondition of second conflict sentence has states which are correct
                for sl, sp in [(l_prec, p_prec)
                               ]:  # Check preconditions and effects
                    for sl_e, sp_e in zip(sl, sp):  # Check all entities
                        for si in [l_conflict[1]]:  # Check conflicting sentences
                            sl_es = sl_e[si]
                            sp_es = sp_e[si]
                            for j, p in enumerate(
                                    sp_es
                            ):  # Check all attributes where there's a nontrivial prediction
                                if p != att_default_values[atts[
                                        j]] and p > 0:  # NOTE: p > 0 is required to avoid counting any padding predictions.
                                    found_states = True
                                    if p != sl_es[j]:
                                        states_verifiable = False

                if states_verifiable and found_states:
                    verifiable += 1
                    explanation['valid_explanation'] = True
                if consistent:
                    consistent_preds += 1
                    explanation['consistent'] = True
        total += 1
        explanations.append(explanation)
                
    return verifiable / total, consistent_preds / total, explanations


def save_results(results, output_dir, dataset_name):
    with open(os.path.join(output_dir, 'results_%s.json' % str(dataset_name)),
              'w') as f:
        json.dump(results, f)
        
        
def trainModel_aug(batch,device,tslm,tslm_optimizer,grad_accmu,loss_abl):
    grad_count=1
    entity_total_number=len(batch['roc_input_1'])+len(batch['roc_input_2'])
    prep_result=torch.tensor([]).to(device)
    effe_result=torch.tensor([]).to(device)
    conflict_result=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}
    story_label=batch['label'][0][0]
    for entity_data in batch['roc_input_1']:
        entity_input=torch.tensor(entity_data['input_ids']).long().to(device)
        entity_mask=torch.tensor(entity_data['attention_mask']).long().to(device)
        entity_timestep=torch.tensor(entity_data['timestamp_id']).long().to(device)
        entity_preconditions=torch.tensor(entity_data['precondition']).long().to(device)
        entity_effects=torch.tensor(entity_data['effect']).long().to(device)
        entity_conflict= torch.tensor(entity_data['conflict']).long().to(device)
        entity_label=torch.tensor([story_label]).long().to(device)
        segment_ids = None
        entity_out=tslm(input_ids=entity_input,attention_mask=entity_mask,timestep_type_ids=entity_timestep\
         ,token_type_ids=segment_ids,prec_label=entity_preconditions,effect_label=entity_effects,\
        conflict=entity_conflict,label=entity_label)

        entity_loss=entity_out['loss_preconditions']*loss_abl[0]+entity_out['loss_effects']*loss_abl[1]\
            +entity_out['loss_conflict']*loss_abl[2]+entity_out['loss_story']*loss_abl[3]
        entity_loss=entity_loss/grad_accmu
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()
        grad_count+=1



        prep_result = torch.cat((prep_result,entity_out['out_preconditions']),dim=0)
        effe_result = torch.cat((effe_result,entity_out['out_effects']),dim=0)
        conflict_result = torch.cat((conflict_result,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']



    for entity_data in batch['roc_input_2']:
        entity_input=torch.tensor(entity_data['input_ids']).long().to(device)
        entity_mask=torch.tensor(entity_data['attention_mask']).long().to(device)
        entity_timestep=torch.tensor(entity_data['timestamp_id']).long().to(device)
        entity_preconditions=torch.tensor(entity_data['precondition']).long().to(device)
        entity_effects=torch.tensor(entity_data['effect']).long().to(device)
        entity_conflict= torch.tensor(entity_data['conflict']).long().to(device)
        entity_label=torch.tensor([1-story_label]).long().to(device)
        segment_ids = None
        entity_out=tslm(input_ids=entity_input,attention_mask=entity_mask,timestep_type_ids=entity_timestep\
         ,token_type_ids=segment_ids,prec_label=entity_preconditions,effect_label=entity_effects,\
        conflict=entity_conflict,label=entity_label)


        entity_loss=entity_out['loss_preconditions']*loss_abl[0]+entity_out['loss_effects']*loss_abl[1]\
            +entity_out['loss_conflict']*loss_abl[2]+entity_out['loss_story']*loss_abl[3]
        entity_loss=entity_loss/grad_accmu
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()
        grad_count+=1



        prep_result = torch.cat((prep_result,entity_out['out_preconditions']),dim=0)
        effe_result = torch.cat((effe_result,entity_out['out_effects']),dim=0)
        conflict_result = torch.cat((conflict_result,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']


    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/entity_total_number
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/entity_total_number
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/entity_total_number
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/entity_total_number
    final_out['total_loss']=total_entity_loss_total/entity_total_number
    
    return final_out