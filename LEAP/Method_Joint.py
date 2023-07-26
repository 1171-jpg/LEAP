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
        
        
        
def train_model_joint(batch,num_attributes,device,tslm,tslm_optimizer,grad_accmu,loss_adjust_joint,loss_adjust_seperate):
    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device) 
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    common_entity=batch[11].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0].view(2,-1)

    # result of the whole batch
    prep_result_1=torch.tensor([]).to(device)
    effe_result_1=torch.tensor([]).to(device)
    conflict_result_1=torch.tensor([]).to(device)
    prep_result_2=torch.tensor([]).to(device)
    effe_result_2=torch.tensor([]).to(device)
    conflict_result_2=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}
    # joint mode
    grad_count=1
    for common_index in range(common_entity):
        entity_input_pair=input_ids[0,:,common_index,:,:].reshape(-1,168)
        entity_mask_pair=input_mask[0,:,common_index,:,:].reshape(-1,168)
        entity_precondition_pair=preconditions[0,:,common_index,:,:].reshape(-1,20)
        entity_effect_pair=effects[0,:,common_index,:,:].reshape(-1,20)
        entity_timestep_pair=timestep_type_ids[0,:,common_index,:,:].reshape(-1,168)
        entity_conflict_pair=conflicts[0,:,common_index,:]
        entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                         ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                        conflict=entity_conflict_pair,label=labels.view(-1),joint_label=True)



        #backward
        entity_loss=entity_out['loss_preconditions']*loss_adjust_joint[0]/20+entity_out['loss_effects']*loss_adjust_joint[1]/20\
            +entity_out['loss_conflict']*loss_adjust_joint[2]+entity_out['loss_story']*loss_adjust_joint[3]
        entity_loss=entity_loss/grad_accmu
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()
        grad_count+=1



        prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions'][0]),dim=0)
        effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects'][0]),dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions'][1]),dim=0)
        effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects'][1]),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']*2
        total_entity_loss_effect+=entity_out['loss_effects']*2
        total_entity_loss_conflicts+=entity_out['loss_conflict']*2
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']

    if grad_count%grad_accmu!=0:
        tslm_optimizer.step()
        tslm_optimizer.zero_grad()    


    #seperate
    grad_count=1
    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[0][entity_index]==1:
            entity_input_single = input_ids[0][0][entity_index]
            entity_mask_single=input_mask[0][0][entity_index]
            entity_precondition_single=preconditions[0][0][entity_index]
            entity_effect_single=effects[0][0][entity_index]
            entity_timestep_single=timestep_type_ids[0][0][entity_index]
            entity_conflict_single=conflicts[0][0][entity_index]
            entity_out = tslm(input_ids=entity_input_single,attention_mask=entity_mask_single,timestep_type_ids=entity_timestep_single\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                            conflict=entity_conflict_single,label=labels.view(-1),joint_label=False)

            entity_loss=entity_out['loss_preconditions']*loss_adjust_seperate[0]/20+entity_out['loss_effects']*loss_adjust_seperate[1]/20\
            +entity_out['loss_conflict']*loss_adjust_seperate[2] 
            entity_loss=entity_loss/grad_accmu
            entity_loss.backward()
            if grad_count%grad_accmu==0:
                tslm_optimizer.step()
                tslm_optimizer.zero_grad()
            grad_count+=1
            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']



    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[1][entity_index]==1:
            entity_input_single = input_ids[0][1][entity_index]
            entity_mask_single=input_mask[0][1][entity_index]
            entity_precondition_single=preconditions[0][1][entity_index]
            entity_effect_single=effects[0][1][entity_index]
            entity_timestep_single=timestep_type_ids[0][1][entity_index]
            entity_conflict_single=conflicts[0][1][entity_index]
            entity_out = tslm(input_ids=entity_input_single,attention_mask=entity_mask_single,timestep_type_ids=entity_timestep_single\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                            conflict=entity_conflict_single,label=labels.view(-1),joint_label=False)

            entity_loss=entity_out['loss_preconditions']*loss_adjust_seperate[0]/20+entity_out['loss_effects']*loss_adjust_seperate[1]/20\
            +entity_out['loss_conflict']*loss_adjust_seperate[2] 
            entity_loss=entity_loss/grad_accmu
            entity_loss.backward()
            if grad_count%grad_accmu==0:
                tslm_optimizer.step()
                tslm_optimizer.zero_grad()
            grad_count+=1
            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']

    # Summarize result
    prep_result = torch.cat((prep_result_1,prep_result_2),dim=0)
    effe_result = torch.cat((effe_result_1,effe_result_2),dim=0)
    conflict_result = torch.cat((conflict_result_1,conflict_result_2),dim=0)
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out={}
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/common_entity.item()
    final_out['total_loss']=total_entity_loss_total/common_entity.item()
    
    return final_out



def eval_model_joint(batch,num_attributes,device,tslm,tslm_optimizer):
    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device) 
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    common_entity=batch[11].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0].view(2,-1)

    # result of the whole batch
    prep_result_1=torch.tensor([]).to(device)
    effe_result_1=torch.tensor([]).to(device)
    conflict_result_1=torch.tensor([]).to(device)
    prep_result_2=torch.tensor([]).to(device)
    effe_result_2=torch.tensor([]).to(device)
    conflict_result_2=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}
    # joint mode
    grad_count=1
    for common_index in range(common_entity):
        entity_input_pair=input_ids[0,:,common_index,:,:].reshape(-1,168)
        entity_mask_pair=input_mask[0,:,common_index,:,:].reshape(-1,168)
        entity_precondition_pair=preconditions[0,:,common_index,:,:].reshape(-1,20)
        entity_effect_pair=effects[0,:,common_index,:,:].reshape(-1,20)
        entity_timestep_pair=timestep_type_ids[0,:,common_index,:,:].reshape(-1,168)
        entity_conflict_pair=conflicts[0,:,common_index,:]
        with torch.no_grad():
            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                            conflict=entity_conflict_pair,label=labels.view(-1),joint_label=True)





        prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions'][0]),dim=0)
        effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects'][0]),dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions'][1]),dim=0)
        effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects'][1]),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']*2
        total_entity_loss_effect+=entity_out['loss_effects']*2
        total_entity_loss_conflicts+=entity_out['loss_conflict']*2
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']



    #seperate
    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[0][entity_index]==1:
            entity_input_single = input_ids[0][0][entity_index]
            entity_mask_single=input_mask[0][0][entity_index]
            entity_precondition_single=preconditions[0][0][entity_index]
            entity_effect_single=effects[0][0][entity_index]
            entity_timestep_single=timestep_type_ids[0][0][entity_index]
            entity_conflict_single=conflicts[0][0][entity_index]
            with torch.no_grad():
                entity_out = tslm(input_ids=entity_input_single,attention_mask=entity_mask_single,timestep_type_ids=entity_timestep_single\
                                 ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                                conflict=entity_conflict_single,label=labels.view(-1),joint_label=False)

            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']



    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[1][entity_index]==1:
            entity_input_single = input_ids[0][1][entity_index]
            entity_mask_single=input_mask[0][1][entity_index]
            entity_precondition_single=preconditions[0][1][entity_index]
            entity_effect_single=effects[0][1][entity_index]
            entity_timestep_single=timestep_type_ids[0][1][entity_index]
            entity_conflict_single=conflicts[0][1][entity_index]
            with torch.no_grad():
                entity_out = tslm(input_ids=entity_input_single,attention_mask=entity_mask_single,timestep_type_ids=entity_timestep_single\
                                 ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                                conflict=entity_conflict_single,label=labels.view(-1),joint_label=False)

            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']

    # Summarize result
    prep_result = torch.cat((prep_result_1,prep_result_2),dim=0)
    effe_result = torch.cat((effe_result_1,effe_result_2),dim=0)
    conflict_result = torch.cat((conflict_result_1,conflict_result_2),dim=0)
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out={}
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/common_entity.item()
    final_out['total_loss']=total_entity_loss_total/common_entity.item()
    
    return final_out





def evaluation_joint_model(max_story_length,num_attributes,tslm,tslm_optimizer,dev_dataloader,device):
    # EVAL
    tslm.zero_grad()
    tslm.eval()


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

        final_out=eval_model_joint(batch,num_attributes,device,tslm,tslm_optimizer)

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
         
        
        story_result = torch.mean(final_out['out_story'],dim=0).to('cpu').numpy()
        if story_result[0]>story_result[1]:
            preds=0
        else:
            preds=1
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



#Three method for dummy joint
def train_model_joint_dummy(batch,num_attributes,device,tslm,tslm_optimizer,grad_accmu,loss_adjust):
    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device) 
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    common_entity=batch[11].long().to(device)
    dummy_input_ids = batch[12].long().to(device)
    dummy_masks = batch[13].long().to(device)
    dummy_timestep_ids = batch[14].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0].view(2,-1)

    # result of the whole batch
    prep_result_1=torch.tensor([]).to(device)
    effe_result_1=torch.tensor([]).to(device)
    conflict_result_1=torch.tensor([]).to(device)
    prep_result_2=torch.tensor([]).to(device)
    effe_result_2=torch.tensor([]).to(device)
    conflict_result_2=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}
    # joint mode
    grad_count=0
    for common_index in range(common_entity):
        maxlength=input_ids.shape[-1]
        entity_input_pair=input_ids[0,:,common_index,:,:].reshape(-1,maxlength)
        entity_mask_pair=input_mask[0,:,common_index,:,:].reshape(-1,maxlength)
        entity_precondition_pair=preconditions[0,:,common_index,:,:].reshape(-1,20)
        entity_effect_pair=effects[0,:,common_index,:,:].reshape(-1,20)
        entity_timestep_pair=timestep_type_ids[0,:,common_index,:,:].reshape(-1,maxlength)
        entity_conflict_pair=conflicts[0,:,common_index,:]
        entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                         ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                        conflict=entity_conflict_pair,label=labels.view(-1),joint_label=0)



        #backward
        entity_loss=entity_out['loss_preconditions']*loss_adjust[0]/20+entity_out['loss_effects']*loss_adjust[1]/20\
            +entity_out['loss_conflict']*loss_adjust[2]+entity_out['loss_story']*loss_adjust[3]
        grad_count+=1
        entity_loss=entity_loss/grad_accmu
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()
        



        prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions'][0]),dim=0)
        effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects'][0]),dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions'][1]),dim=0)
        effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects'][1]),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']*2
        total_entity_loss_effect+=entity_out['loss_effects']*2
        total_entity_loss_conflicts+=entity_out['loss_conflict']*2
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']
 

    #seperate
    #story 1
    grad_count=1
    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[0][entity_index]==1:
            entity_input_single = input_ids[0][0][entity_index]
            entity_mask_single=input_mask[0][0][entity_index]
            entity_precondition_single=preconditions[0][0][entity_index]
            entity_effect_single=effects[0][0][entity_index]
            entity_timestep_single=timestep_type_ids[0][0][entity_index]
            entity_conflict_single=conflicts[0][0][entity_index]
            dummy_input_single=dummy_input_ids[0][1]
            dummy_mask_single=dummy_masks[0][1]
            dummy_time_single=dummy_timestep_ids[0][1]

            entity_input_pair=torch.cat((entity_input_single,dummy_input_single),dim=0)
            entity_mask_pair=torch.cat((entity_mask_single,dummy_mask_single),dim=0)
            entity_timestep_pair=torch.cat((entity_timestep_single,dummy_time_single),dim=0)

            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                            conflict=entity_conflict_single,label=labels.view(-1),joint_label=1)

            entity_loss=entity_out['loss_preconditions']*loss_adjust[0]/20+entity_out['loss_effects']*loss_adjust[1]/20\
            +entity_out['loss_conflict']*loss_adjust[2]+entity_out['loss_story']*loss_adjust[3]
            
            grad_count+=1
            entity_loss=entity_loss/grad_accmu
            entity_loss.backward()
            if grad_count%grad_accmu==0:
                tslm_optimizer.step()
                tslm_optimizer.zero_grad()
            
            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
            story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
            total_entity_loss_stories+=entity_out['loss_story']
            total_entity_loss_total+=entity_out['total_loss']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)





   #story 2
    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[1][entity_index]==1:
            entity_input_single = input_ids[0][1][entity_index]
            entity_mask_single=input_mask[0][1][entity_index]
            entity_precondition_single=preconditions[0][1][entity_index]
            entity_effect_single=effects[0][1][entity_index]
            entity_timestep_single=timestep_type_ids[0][1][entity_index]
            entity_conflict_single=conflicts[0][1][entity_index]
            dummy_input_single=dummy_input_ids[0][0]
            dummy_mask_single=dummy_masks[0][0]
            dummy_time_single=dummy_timestep_ids[0][0]

            entity_input_pair=torch.cat((dummy_input_single,entity_input_single),dim=0)
            entity_mask_pair=torch.cat((dummy_mask_single,entity_mask_single),dim=0)
            entity_timestep_pair=torch.cat((dummy_time_single,entity_timestep_single),dim=0)

            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                            conflict=entity_conflict_single,label=labels.view(-1),joint_label=2)

            entity_loss=entity_out['loss_preconditions']*loss_adjust[0]/20+entity_out['loss_effects']*loss_adjust[1]/20\
            +entity_out['loss_conflict']*loss_adjust[2]+entity_out['loss_story']*loss_adjust[3]
            
            grad_count+=1
            entity_loss=entity_loss/grad_accmu
            entity_loss.backward()
            if grad_count%grad_accmu==0:
                tslm_optimizer.step()
                tslm_optimizer.zero_grad()

            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
            story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
            total_entity_loss_stories+=entity_out['loss_story']
            total_entity_loss_total+=entity_out['total_loss']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)

    if grad_count%grad_accmu!=0:
        tslm_optimizer.step()
        tslm_optimizer.zero_grad() 

    # Summarize result
    prep_result = torch.cat((prep_result_1,prep_result_2),dim=0)
    effe_result = torch.cat((effe_result_1,effe_result_2),dim=0)
    conflict_result = torch.cat((conflict_result_1,conflict_result_2),dim=0)
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out={}
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/len(story_result)
    final_out['total_loss']=total_entity_loss_total/len(story_result)
    
    return final_out



def eval_model_joint_dummy(batch,num_attributes,device,tslm,tslm_optimizer):
    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device) 
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids = batch[9].long().to(device)
    all_entity_labels=batch[10].long().to(device)
    common_entity=batch[11].long().to(device)
    dummy_input_ids = batch[12].long().to(device)
    dummy_masks = batch[13].long().to(device)
    dummy_timestep_ids = batch[14].long().to(device)
    segment_ids = None
    lengthmask=getLengthMask(input_ids,input_lengths,input_entities)
    active_entity_list=lengthmask[:,0].view(2,-1)

    # result of the whole batch
    prep_result_1=torch.tensor([]).to(device)
    effe_result_1=torch.tensor([]).to(device)
    conflict_result_1=torch.tensor([]).to(device)
    prep_result_2=torch.tensor([]).to(device)
    effe_result_2=torch.tensor([]).to(device)
    conflict_result_2=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    final_out={}
    # joint mode
    grad_count=1
    for common_index in range(common_entity):
        entity_input_pair=input_ids[0,:,common_index,:,:].reshape(-1,168)
        entity_mask_pair=input_mask[0,:,common_index,:,:].reshape(-1,168)
        entity_precondition_pair=preconditions[0,:,common_index,:,:].reshape(-1,20)
        entity_effect_pair=effects[0,:,common_index,:,:].reshape(-1,20)
        entity_timestep_pair=timestep_type_ids[0,:,common_index,:,:].reshape(-1,168)
        entity_conflict_pair=conflicts[0,:,common_index,:]
        with torch.no_grad():
            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                            conflict=entity_conflict_pair,label=labels.view(-1),joint_label=0)



        prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions'][0]),dim=0)
        effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects'][0]),dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions'][1]),dim=0)
        effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects'][1]),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']*2
        total_entity_loss_effect+=entity_out['loss_effects']*2
        total_entity_loss_conflicts+=entity_out['loss_conflict']*2
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']



    #seperate
    #story 1
    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[0][entity_index]==1:
            entity_input_single = input_ids[0][0][entity_index]
            entity_mask_single=input_mask[0][0][entity_index]
            entity_precondition_single=preconditions[0][0][entity_index]
            entity_effect_single=effects[0][0][entity_index]
            entity_timestep_single=timestep_type_ids[0][0][entity_index]
            entity_conflict_single=conflicts[0][0][entity_index]
            dummy_input_single=dummy_input_ids[0][1]
            dummy_mask_single=dummy_masks[0][1]
            dummy_time_single=dummy_timestep_ids[0][1]

            entity_input_pair=torch.cat((entity_input_single,dummy_input_single),dim=0)
            entity_mask_pair=torch.cat((entity_mask_single,dummy_mask_single),dim=0)
            entity_timestep_pair=torch.cat((entity_timestep_single,dummy_time_single),dim=0)
            with torch.no_grad():
                entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                                 ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                                conflict=entity_conflict_single,label=labels.view(-1),joint_label=1)

            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
            story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
            total_entity_loss_stories+=entity_out['loss_story']
            total_entity_loss_total+=entity_out['total_loss']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
            effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
            conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)





   #story 2
    for entity_index in range(common_entity[0],active_entity_list.shape[1]):
        if active_entity_list[1][entity_index]==1:
            entity_input_single = input_ids[0][1][entity_index]
            entity_mask_single=input_mask[0][1][entity_index]
            entity_precondition_single=preconditions[0][1][entity_index]
            entity_effect_single=effects[0][1][entity_index]
            entity_timestep_single=timestep_type_ids[0][1][entity_index]
            entity_conflict_single=conflicts[0][1][entity_index]
            dummy_input_single=dummy_input_ids[0][0]
            dummy_mask_single=dummy_masks[0][0]
            dummy_time_single=dummy_timestep_ids[0][0]

            entity_input_pair=torch.cat((dummy_input_single,entity_input_single),dim=0)
            entity_mask_pair=torch.cat((dummy_mask_single,entity_mask_single),dim=0)
            entity_timestep_pair=torch.cat((dummy_time_single,entity_timestep_single),dim=0)
            with torch.no_grad():
                entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                                 ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                                conflict=entity_conflict_single,label=labels.view(-1),joint_label=2)

            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
            story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)
            total_entity_loss_preconditions+=entity_out['loss_preconditions']
            total_entity_loss_effect+=entity_out['loss_effects']
            total_entity_loss_conflicts+=entity_out['loss_conflict']
            total_entity_loss_stories+=entity_out['loss_story']
            total_entity_loss_total+=entity_out['total_loss']
        else:
            entity_out={'out_preconditions': torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_effects' : torch.zeros(preconditions[0][0][0].shape).to(device),
             'out_conflict': torch.zeros(int(len(preconditions[0][0][0])*(len(preconditions[0][0][0])-1)/2)).to(device),
             'out_story' : torch.zeros([2]).to(device),
             'loss_preconditions': 0,
             'loss_effects': 0, 
             'loss_conflict' : 0,
             'loss_story' : 0,
             'total_loss' : 0}
            prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
            effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
            conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)



    # Summarize result
    prep_result = torch.cat((prep_result_1,prep_result_2),dim=0)
    effe_result = torch.cat((effe_result_1,effe_result_2),dim=0)
    conflict_result = torch.cat((conflict_result_1,conflict_result_2),dim=0)
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    final_out={}
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/torch.sum(input_entities).item()
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/torch.sum(input_entities).item()
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/torch.sum(input_entities).item()
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/len(story_result)
    final_out['total_loss']=total_entity_loss_total/len(story_result)
    
    return final_out



def evaluation_joint_model_dummy(max_story_length,num_attributes,tslm,tslm_optimizer,dev_dataloader,device,sentence_tag):
    # EVAL
    tslm.zero_grad()
    tslm.eval()


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

        final_out=eval_model_joint_dummy(batch,num_attributes,device,tslm,tslm_optimizer)

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
         
        if sentence_tag:
            conflict_result=final_out['out_conflicts'].view(2,-1,final_out['out_conflicts'].shape[-1])
            story_out= -torch.sum(conflict_result, dim=(1,2))
            story_out[0]/=input_entities[0][0]  # average over entity number
            story_out[1]/=input_entities[0][1]
            story_result=torch.mean(final_out['out_story'],dim=0).detach().to('cpu').numpy()
            preds=torch.argmax(story_out).view(-1,1).to('cpu').numpy()
        else:
            story_result = torch.mean(final_out['out_story'],dim=0).to('cpu').numpy()
            if story_result[0]>story_result[1]:
                preds=0
            else:
                preds=1
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




def train_model_joint_dummy_cskg(batch,num_attributes,device,tslm,tslm_optimizer,grad_accmu,loss_adjust):

    entity_total_number=len(batch['entity_1'])+len(batch['entity_2'])
    # result of the whole batch
    prep_result_1=torch.tensor([]).to(device)
    effe_result_1=torch.tensor([]).to(device)
    conflict_result_1=torch.tensor([]).to(device)
    prep_result_2=torch.tensor([]).to(device)
    effe_result_2=torch.tensor([]).to(device)
    conflict_result_2=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_total=0
    segment_ids = None
    final_out={}
    story_label=batch['label']
    entity_pair_label=torch.tensor([story_label]).long().to(device)

    # joint mode
    grad_count=0
    for common_index in range(batch['common_entity']):
        entity_input_pair=torch.cat((\
                                     torch.tensor(batch['cskg_input_1'][common_index]['input_ids']).long().to(device),\
                                     torch.tensor(batch['cskg_input_2'][common_index]['input_ids']).long().to(device)),dim=0)
        entity_mask_pair=torch.cat((\
                             torch.tensor(batch['cskg_input_1'][common_index]['attention_mask']).long().to(device),\
                             torch.tensor(batch['cskg_input_2'][common_index]['attention_mask']).long().to(device)),dim=0)
        entity_precondition_pair=torch.cat((\
                             torch.tensor(batch['cskg_input_1'][common_index]['precondition']).long().to(device),\
                             torch.tensor(batch['cskg_input_2'][common_index]['precondition']).long().to(device)),dim=0)
        entity_effect_pair=torch.cat((\
                             torch.tensor(batch['cskg_input_1'][common_index]['effect']).long().to(device),\
                             torch.tensor(batch['cskg_input_2'][common_index]['effect']).long().to(device)),dim=0)
        entity_timestep_pair=torch.cat((\
                             torch.tensor(batch['cskg_input_1'][common_index]['timestamp_id']).long().to(device),\
                             torch.tensor(batch['cskg_input_2'][common_index]['timestamp_id']).long().to(device)),dim=0)
        entity_conflict_pair=torch.cat((\
                             torch.tensor([batch['cskg_input_1'][common_index]['conflict']]).long().to(device),\
                             torch.tensor([batch['cskg_input_2'][common_index]['conflict']]).long().to(device)),dim=0)
        entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                         ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                        conflict=entity_conflict_pair,label=entity_pair_label,joint_label=0)



        #backward
        grad_count+=1
#         entity_loss=entity_out['loss_story']*loss_adjust[3]/grad_accmu
        entity_loss=entity_out['loss_preconditions']*loss_adjust[0]/20+entity_out['loss_effects']*loss_adjust[1]/20\
            +entity_out['loss_conflict']*loss_adjust[2]+entity_out['loss_story']*loss_adjust[3]
        entity_loss=entity_loss/grad_accmu
        
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()



        prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions'][0]),dim=0)
        effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects'][0]),dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions'][1]),dim=0)
        effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects'][1]),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)    
        total_entity_loss_preconditions+=entity_out['loss_preconditions']*2
        total_entity_loss_effect+=entity_out['loss_effects']*2
        total_entity_loss_conflicts+=entity_out['loss_conflict']*2
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']



    #seperate
    #story 1
    for entity_index in range(batch['common_entity'],len(batch['entity_1'])):
        entity_input_pair=torch.cat((\
                                     torch.tensor(batch['cskg_input_1'][entity_index]['input_ids']).long().to(device),\
                                     torch.tensor(batch['cskg_dummy_2']['input_ids']).long().to(device)),dim=0)
        entity_mask_pair=torch.cat((\
                             torch.tensor(batch['cskg_input_1'][entity_index]['attention_mask']).long().to(device),\
                             torch.tensor(batch['cskg_dummy_2']['attention_mask']).long().to(device)),dim=0)
        entity_timestep_pair=torch.cat((\
                             torch.tensor(batch['cskg_input_1'][entity_index]['timestamp_id']).long().to(device),\
                             torch.tensor(batch['cskg_dummy_2']['timestamp_id']).long().to(device)),dim=0)
        entity_precondition_single=torch.tensor(batch['cskg_input_1'][entity_index]['precondition']).long().to(device)
        entity_effect_single=torch.tensor(batch['cskg_input_1'][entity_index]['effect']).long().to(device) 
        entity_conflict_single=torch.tensor(batch['cskg_input_1'][entity_index]['conflict']).long().to(device)
        entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                         ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                        conflict=entity_conflict_single,label=entity_pair_label,joint_label=1)

        #backward
        grad_count+=1
        entity_loss=entity_out['loss_preconditions']*loss_adjust[0]/20+entity_out['loss_effects']*loss_adjust[1]/20\
            +entity_out['loss_conflict']*loss_adjust[2]+entity_out['loss_story']*loss_adjust[3]
        entity_loss=entity_loss/grad_accmu
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()

        prep_result_1 = torch.cat((prep_result_1,entity_out['out_preconditions']),dim=0)
        effe_result_1 = torch.cat((effe_result_1,entity_out['out_effects']),dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']




   #story 2
    for entity_index in range(batch['common_entity'],len(batch['entity_2'])):
        entity_input_pair=torch.cat((\
                                     torch.tensor(batch['cskg_dummy_1']['input_ids']).long().to(device),\
                                     torch.tensor(batch['cskg_input_2'][entity_index]['input_ids']).long().to(device)),dim=0)
        entity_mask_pair=torch.cat((\
                             torch.tensor(batch['cskg_dummy_1']['attention_mask']).long().to(device),\
                             torch.tensor(batch['cskg_input_2'][entity_index]['attention_mask']).long().to(device)),dim=0)
        entity_timestep_pair=torch.cat((\
                             torch.tensor(batch['cskg_dummy_1']['timestamp_id']).long().to(device),\
                             torch.tensor(batch['cskg_input_2'][entity_index]['timestamp_id']).long().to(device)),dim=0)
        entity_precondition_single=torch.tensor(batch['cskg_input_2'][entity_index]['precondition']).long().to(device)
        entity_effect_single=torch.tensor(batch['cskg_input_2'][entity_index]['effect']).long().to(device) 
        entity_conflict_single=torch.tensor(batch['cskg_input_2'][entity_index]['conflict']).long().to(device)
        entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                         ,token_type_ids=segment_ids,prec_label=entity_precondition_single,effect_label=entity_effect_single,\
                        conflict=entity_conflict_single,label=entity_pair_label,joint_label=2)

        #backward
        grad_count+=1
#         entity_loss=entity_out['loss_story']*loss_adjust[3]/grad_accmu
        entity_loss=entity_out['loss_preconditions']*loss_adjust[0]/20+entity_out['loss_effects']*loss_adjust[1]/20\
            +entity_out['loss_conflict']*loss_adjust[2]+entity_out['loss_story']*loss_adjust[3]
        entity_loss=entity_loss/grad_accmu
        entity_loss.backward()
        if grad_count%grad_accmu==0:
            tslm_optimizer.step()
            tslm_optimizer.zero_grad()

        prep_result_2 = torch.cat((prep_result_2,entity_out['out_preconditions']),dim=0)
        effe_result_2 = torch.cat((effe_result_2,entity_out['out_effects']),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)
        story_result = torch.cat((story_result,entity_out['out_story'].view(1,-1)),dim=0)
        total_entity_loss_preconditions+=entity_out['loss_preconditions']
        total_entity_loss_effect+=entity_out['loss_effects']
        total_entity_loss_conflicts+=entity_out['loss_conflict']
        total_entity_loss_stories+=entity_out['loss_story']
        total_entity_loss_total+=entity_out['total_loss']
                                  
    if grad_count%grad_accmu!=0:
        tslm_optimizer.step()
        tslm_optimizer.zero_grad() 

    # Summarize result
    prep_result = torch.cat((prep_result_1,prep_result_2),dim=0)
    effe_result = torch.cat((effe_result_1,effe_result_2),dim=0)
    conflict_result = torch.cat((conflict_result_1,conflict_result_2),dim=0)

    final_out={}
    final_out['out_preconditions']=prep_result
    final_out['loss_preconditions']=total_entity_loss_preconditions/entity_total_number
    final_out['out_effects']=effe_result
    final_out['loss_effects']=total_entity_loss_effect/entity_total_number
    final_out['out_conflicts']=conflict_result
    final_out['loss_conflicts']=total_entity_loss_conflicts/entity_total_number
    final_out['out_story']=story_result
    final_out['loss_story']=total_entity_loss_stories/len(story_result)
    final_out['total_loss']=total_entity_loss_total/len(story_result)
    
    return final_out