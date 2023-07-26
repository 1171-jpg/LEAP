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
    
    
def orderdata(dev_dataset):
    for sample in dev_dataset:
        list1=sorted(sample['entity_1'])
        list2=sorted(sample['entity_2'])
        common_list=set(list1)&set(list2)
        common_list=sorted(list(common_list))
        entity_list1=[]
        entity_list2=[]
        sample['common_entity']=len(common_list)
        for entity in common_list:
            entity_list1.append(entity)
            entity_list2.append(entity)
        list1_left=set(list1)-set(common_list)
        list2_left=set(list2)-set(common_list)
        for item in list1_left:
            entity_list1.append(item)
        for item in list2_left:
            entity_list2.append(item)
        sample['entity_1']=entity_list1
        sample['entity_2']=entity_list2
        
    return dev_dataset 



def add_input_feature_joint_soft_dummy(dataset,tokenizer,maxStoryLength):
    print("Add input feature joint")
    for sample_data in tqdm(dataset):
        anli_input_1=[]
        for entity in sample_data['entity_1']:           

            question="Where is "+str(entity)+"?! </s> "
            story=""
            for idx,sentence in enumerate(sample_data['goal_sol_1']):
                story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_1'])-1 else story+sentence
            qaStories=[question+story]*len(sample_data['goal_sol_1'])
            inputs=tokenizer(qaStories)
            f_out=[]
            for time in range(len(sample_data['goal_sol_1'])):
                timestamp_id=[]
                check= -1
                for index,ids in enumerate(inputs['input_ids'][0]):
                    if ids == 2:
                        check += 1
                    if check == -1:
                        timestamp_id.append(0)
                    elif ids == 2:
                        timestamp_id.append(0)
                    else:
                        if check < time :
                            timestamp_id.append(1)
                        elif check == time:
                            timestamp_id.append(2)
                        else:
                            timestamp_id.append(3)
                f_out.append(timestamp_id)
            input_ids=inputs['input_ids']
            attention_mask=inputs["attention_mask"]
            for index in range(len(input_ids)):
                assert len(input_ids[index]) <= maxStoryLength
                paddingLength=maxStoryLength-len(input_ids[index])
                input_ids[index]=input_ids[index]+[0]*paddingLength
                attention_mask[index]=attention_mask[index]+[0]*paddingLength
                f_out[index]=f_out[index]+[0]*paddingLength           
            anli_input_1.append({'input_ids':torch.tensor(input_ids),
                                'attention_mask':torch.tensor(attention_mask),
                                'timestamp_id':torch.tensor(f_out)})
        # dummy input 1
        question="Where is _?! </s> "
        story=""
        for idx,sentence in enumerate(sample_data['goal_sol_1']):
            story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_1'])-1 else story+sentence
        qaStories=[question+story]*len(sample_data['goal_sol_1'])
        inputs=tokenizer(qaStories)
        f_out=[]
        for time in range(len(sample_data['goal_sol_1'])):
            timestamp_id=[]
            check= -1
            for index,ids in enumerate(inputs['input_ids'][0]):
                if ids == 2:
                    check += 1
                if check == -1:
                    timestamp_id.append(0)
                elif ids == 2:
                    timestamp_id.append(0)
                else:
                    if check < time :
                        timestamp_id.append(1)
                    elif check == time:
                        timestamp_id.append(2)
                    else:
                        timestamp_id.append(3)
            f_out.append(timestamp_id)
        input_ids=inputs['input_ids']
        attention_mask=inputs["attention_mask"]
        for index in range(len(input_ids)):
            assert len(input_ids[index]) <= maxStoryLength
            paddingLength=maxStoryLength-len(input_ids[index])
            input_ids[index]=input_ids[index]+[0]*paddingLength
            attention_mask[index]=attention_mask[index]+[0]*paddingLength
            f_out[index]=f_out[index]+[0]*paddingLength 
        sample_data['anli_dummy_1']={'input_ids':torch.tensor(input_ids),
                            'attention_mask':torch.tensor(attention_mask),
                            'timestamp_id':torch.tensor(f_out)}     
    

        anli_input_2=[]
        for entity in sample_data['entity_2']:
            
            question="Where is "+str(entity)+"?! </s> "
            story=""
            for idx,sentence in enumerate(sample_data['goal_sol_2']):
                story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_2'])-1 else story+sentence
            qaStories=[question+story]*len(sample_data['goal_sol_2'])
            inputs=tokenizer(qaStories)
            f_out=[]
            for time in range(len(sample_data['goal_sol_2'])):
                timestamp_id=[]
                check= -1
                for index,ids in enumerate(inputs['input_ids'][0]):
                    if ids == 2:
                        check += 1
                    if check == -1:
                        timestamp_id.append(0)
                    elif ids == 2:
                        timestamp_id.append(0)
                    else:
                        if check < time :
                            timestamp_id.append(1)
                        elif check == time:
                            timestamp_id.append(2)
                        else:
                            timestamp_id.append(3)
                f_out.append(timestamp_id)
            input_ids=inputs['input_ids']
            attention_mask=inputs["attention_mask"]
            for index in range(len(input_ids)):
                assert len(input_ids[index]) <= maxStoryLength
                paddingLength=maxStoryLength-len(input_ids[index])
                input_ids[index]=input_ids[index]+[0]*paddingLength
                attention_mask[index]=attention_mask[index]+[0]*paddingLength
                f_out[index]=f_out[index]+[0]*paddingLength 
            anli_input_2.append({'input_ids':torch.tensor(input_ids),
                                'attention_mask':torch.tensor(attention_mask),
                                'timestamp_id':torch.tensor(f_out)})
            
        # dummy input 2
        question="Where is _?! </s> "
        story=""
        for idx,sentence in enumerate(sample_data['goal_sol_2']):
            story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_2'])-1 else story+sentence
        qaStories=[question+story]*len(sample_data['goal_sol_2'])
        inputs=tokenizer(qaStories)
        f_out=[]
        for time in range(len(sample_data['goal_sol_2'])):
            timestamp_id=[]
            check= -1
            for index,ids in enumerate(inputs['input_ids'][0]):
                if ids == 2:
                    check += 1
                if check == -1:
                    timestamp_id.append(0)
                elif ids == 2:
                    timestamp_id.append(0)
                else:
                    if check < time :
                        timestamp_id.append(1)
                    elif check == time:
                        timestamp_id.append(2)
                    else:
                        timestamp_id.append(3)
            f_out.append(timestamp_id)
        input_ids=inputs['input_ids']
        attention_mask=inputs["attention_mask"]
        for index in range(len(input_ids)):
            assert len(input_ids[index]) <= maxStoryLength
            paddingLength=maxStoryLength-len(input_ids[index])
            input_ids[index]=input_ids[index]+[0]*paddingLength
            attention_mask[index]=attention_mask[index]+[0]*paddingLength
            f_out[index]=f_out[index]+[0]*paddingLength 
        sample_data['anli_dummy_2']={'input_ids':torch.tensor(input_ids),
                            'attention_mask':torch.tensor(attention_mask),
                            'timestamp_id':torch.tensor(f_out)}  
            
            
        sample_data['anli_input_1']=anli_input_1
        sample_data['anli_input_2']=anli_input_2
    return dataset



def predict_from_zero_shot_dummy(tslm,batch,device,sentence_tag = None):
    input_id_list_1 = batch['anli_input_1']
    input_duumy_1= batch['anli_dummy_1']
    entity_number_1 = len(input_id_list_1)
    input_id_list_2 = batch['anli_input_2']
    input_duumy_2= batch['anli_dummy_2']
    entity_number_2 = len(input_id_list_2)    
    label = batch['label']
    conflict_result_1=torch.tensor([]).to(device)
    conflict_result_2=torch.tensor([]).to(device)
    story_result=torch.tensor([]).to(device)
    #common
    for entity_index in range(batch['common_entity']):
        entity_input_pair=torch.cat(\
                                    (input_id_list_1[entity_index]['input_ids'].long().to(device),\
                                     input_id_list_2[entity_index]['input_ids'].long().to(device)),dim=0)
        entity_mask_pair=torch.cat(\
                                    (input_id_list_1[entity_index]['attention_mask'].long().to(device),\
                                     input_id_list_2[entity_index]['attention_mask'].long().to(device)),dim=0)
        entity_timestep_pair=torch.cat(\
                                    (input_id_list_1[entity_index]['timestamp_id'].long().to(device),\
                                     input_id_list_2[entity_index]['timestamp_id'].long().to(device)),dim=0)
        
        entity_precondition_pair = torch.zeros([len(entity_input_pair),20]).long().to(device)
        entity_effect_pair = torch.zeros([len(entity_input_pair), 20]).long().to(device)
        entity_conflict_pair = torch.zeros([2,len(entity_input_pair)]).long().to(device)
        entity_label = torch.tensor([0]).long().to(device)
        segment_ids = None
        with torch.no_grad():
            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                            conflict=entity_conflict_pair,label=entity_label,joint_label=0)
        story_result = torch.cat((story_result, entity_out['out_story'].view(1, -1)), dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
    for entity_index in range(batch['common_entity'],len(batch['entity_1'])):
        entity_input_pair=torch.cat(\
                                    (input_id_list_1[entity_index]['input_ids'].long().to(device),\
                                     input_duumy_2['input_ids'].long().to(device)),dim=0)
        entity_mask_pair=torch.cat(\
                                    (input_id_list_1[entity_index]['attention_mask'].long().to(device),\
                                     input_duumy_2['attention_mask'].long().to(device)),dim=0)
        entity_timestep_pair=torch.cat(\
                                    (input_id_list_1[entity_index]['timestamp_id'].long().to(device),\
                                     input_duumy_2['timestamp_id'].long().to(device)),dim=0)
        
        entity_precondition_pair = torch.zeros([len(input_id_list_1[entity_index]['input_ids']),20]).long().to(device)
        entity_effect_pair = torch.zeros([len(input_id_list_1[entity_index]['input_ids']), 20]).long().to(device)
        entity_conflict_pair = torch.zeros([len(input_id_list_1[entity_index]['input_ids'])]).long().to(device)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'].view(1,-1)),dim=0)
        entity_label = torch.tensor([0]).long().to(device)
        segment_ids = None
        with torch.no_grad():
            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                            conflict=entity_conflict_pair,label=entity_label,joint_label=1)
        story_result = torch.cat((story_result, entity_out['out_story'].view(1, -1)), dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
    for entity_index in range(batch['common_entity'],len(batch['entity_2'])):
        entity_input_pair=torch.cat(\
                                    (input_duumy_1['input_ids'].long().to(device),\
                                     input_id_list_2[entity_index]['input_ids'].long().to(device)),dim=0)
        entity_mask_pair=torch.cat(\
                                    (input_duumy_1['attention_mask'].long().to(device),\
                                     input_id_list_2[entity_index]['attention_mask'].long().to(device)),dim=0)
        entity_timestep_pair=torch.cat(\
                                    (input_duumy_1['timestamp_id'].long().to(device),\
                                     input_id_list_2[entity_index]['timestamp_id'].long().to(device)),dim=0)
        
        entity_precondition_pair = torch.zeros([len(input_id_list_2[entity_index]['input_ids']),20]).long().to(device)
        entity_effect_pair = torch.zeros([len(input_id_list_2[entity_index]['input_ids']), 20]).long().to(device)
        entity_conflict_pair = torch.zeros([len(input_id_list_2[entity_index]['input_ids'])]).long().to(device)
        entity_label = torch.tensor([0]).long().to(device)
        segment_ids = None
        with torch.no_grad():
            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                            conflict=entity_conflict_pair,label=entity_label,joint_label=2)
        story_result = torch.cat((story_result, entity_out['out_story'].view(1, -1)), dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'].view(1,-1)),dim=0)

    if entity_number_1+entity_number_2 ==0:
        entity_input_pair=torch.cat(\
                                    (input_duumy_1['input_ids'].long().to(device),\
                                     input_duumy_2['input_ids'].long().to(device)),dim=0)
        entity_mask_pair=torch.cat(\
                                    (input_duumy_1['attention_mask'].long().to(device),\
                                     input_duumy_2['attention_mask'].long().to(device)),dim=0)
        entity_timestep_pair=torch.cat(\
                                    (input_duumy_1['timestamp_id'].long().to(device),\
                                     input_duumy_2['timestamp_id'].long().to(device)),dim=0)
        
        entity_precondition_pair = torch.zeros([len(entity_input_pair),20]).long().to(device)
        entity_effect_pair = torch.zeros([len(entity_input_pair), 20]).long().to(device)
        entity_conflict_pair = torch.zeros([2,len(entity_input_pair)]).long().to(device)
        entity_label = torch.tensor([0]).long().to(device)
        segment_ids = None
        with torch.no_grad():
            entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                             ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                            conflict=entity_conflict_pair,label=entity_label,joint_label=0)
        story_result = torch.cat((story_result, entity_out['out_story'].view(1, -1)), dim=0)
        conflict_result_1 = torch.cat((conflict_result_1,entity_out['out_conflict'][0].view(1,-1)),dim=0)
        conflict_result_2 = torch.cat((conflict_result_2,entity_out['out_conflict'][1].view(1,-1)),dim=0)
    data_summary={}
    conflict_result = torch.cat((conflict_result_1,conflict_result_2),dim=0)
    if sentence_tag:
        story_out = torch.zeros(2)
        story_out[0]=-torch.sum(conflict_result_1)/entity_number_1
        story_out[1]=-torch.sum(conflict_result_2)/entity_number_2
        sol_pred=torch.argmax(story_out).view(-1,1).to('cpu').numpy()
    else:
        story_out=np.mean(story_result.cpu().numpy(),axis=0)
        sol_pred = 0 if story_out[0]> story_out[1] else 1
        sol_pred = np.array([[sol_pred]])
    data_summary['conflict_result'] = conflict_result
    data_summary['story_result']=story_result.tolist()
    data_summary['story_out']=story_out.tolist()
    data_summary['label_pred']=sol_pred.tolist()
    data_summary['true_label']=label
    
    return sol_pred,data_summary




def find_best_model(FILE_PATH,task,epoch_num,all_flag = False):
    best_accuracy=0
    model1=0
    best_consistency=0
    model2=0
    best_verifiablity=0
    model3=0
    for i in range(epoch_num):
        Epoch_number=str(i)
        result_file_name="results_trip_stories_{}.json".format(task)
        result_file_path=os.path.join(FILE_PATH,Epoch_number,result_file_name)
        result=json.loads(open(result_file_path, 'r').read())
        if result['accuracy']>=best_accuracy:
            best_accuracy=result['accuracy']
            model1=i
        if result['consistency']>best_consistency:
            best_consistency=result['consistency']
            model2=i
        if result['verifiability']>best_verifiablity:
            best_verifiablity=result['verifiability']
            model3=i

    print("Achieve best accuracy {} at Epoch {}".format(best_accuracy,model1))
    print("Achieve best consistency {} at Epoch {}".format(best_consistency,model2))
    print("Achieve best verifiablity {} at Epoch {}".format(best_verifiablity,model3))
    
    if all_flag:
        return model1,best_accuracy,best_consistency,best_verifiablity
    else:
        return model1
    
    
    
def add_input_feature_joint_soft_dummy_codah(dataset,tokenizer,maxStoryLength):
    print("Add input feature joint")
    for sample_data in tqdm(dataset):
        codah_input=[]
        dummy_input=[]
        for entity_total,sentence_total in zip(sample_data['entity'],sample_data['sentences']):
            temp_codah_input=[]
            for entity in entity_total:
                question="Where is "+str(entity)+"?! </s> "
                story =""
                for idx,sentence in enumerate(sentence_total):
                    story=story+sentence+" </s> " if idx< len(sentence_total)-1 else story+sentence            
                qaStories=[question+story]*len(sentence_total)        
                inputs=tokenizer(qaStories)
                f_out=[]
                for time in range(len(sentence_total)):
                    timestamp_id=[]
                    check= -1
                    for index,ids in enumerate(inputs['input_ids'][0]):
                        if ids == 2:
                            check += 1
                        if check == -1:
                            timestamp_id.append(0)
                        elif ids == 2:
                            timestamp_id.append(0)
                        else:
                            if check < time :
                                timestamp_id.append(1)
                            elif check == time:
                                timestamp_id.append(2)
                            else:
                                timestamp_id.append(3)
                    f_out.append(timestamp_id)
                input_ids=inputs['input_ids']
                attention_mask=inputs["attention_mask"]
                for index in range(len(input_ids)):
                    assert len(input_ids[index]) <= maxStoryLength
                    paddingLength=maxStoryLength-len(input_ids[index])
                    input_ids[index]=input_ids[index]+[0]*paddingLength
                    attention_mask[index]=attention_mask[index]+[0]*paddingLength
                    f_out[index]=f_out[index]+[0]*paddingLength           
                temp_codah_input.append({'input_ids':torch.tensor(input_ids),
                                    'attention_mask':torch.tensor(attention_mask),
                                    'timestamp_id':torch.tensor(f_out)})     
            codah_input.append(temp_codah_input)
        
        
        # dummy input
            question="Where is _?! </s> "
            story=""
            for idx,sentence in enumerate(sentence_total):
                story=story+sentence+" </s> " if idx< len(sentence_total)-1 else story+sentence
            qaStories=[question+story]*len(sentence_total)
            inputs=tokenizer(qaStories)
            f_out=[]
            for time in range(len(sentence_total)):
                timestamp_id=[]
                check= -1
                for index,ids in enumerate(inputs['input_ids'][0]):
                    if ids == 2:
                        check += 1
                    if check == -1:
                        timestamp_id.append(0)
                    elif ids == 2:
                        timestamp_id.append(0)
                    else:
                        if check < time :
                            timestamp_id.append(1)
                        elif check == time:
                            timestamp_id.append(2)
                        else:
                            timestamp_id.append(3)
                f_out.append(timestamp_id)
            input_ids=inputs['input_ids']
            attention_mask=inputs["attention_mask"]
            for index in range(len(input_ids)):
                assert len(input_ids[index]) <= maxStoryLength
                paddingLength=maxStoryLength-len(input_ids[index])
                input_ids[index]=input_ids[index]+[0]*paddingLength
                attention_mask[index]=attention_mask[index]+[0]*paddingLength
                f_out[index]=f_out[index]+[0]*paddingLength 
            dummy_input.append({'input_ids':torch.tensor(input_ids),
                                'attention_mask':torch.tensor(attention_mask),
                                'timestamp_id':torch.tensor(f_out)}) 


        sample_data["codah_input"]=codah_input
        sample_data["dummy_input"]=dummy_input
    return dataset



def predict_from_zero_shot_dummy_codah(tslm,batch,device,sentence_tag = None):
    
    codah_input = batch['codah_input']
    dummy_input = batch['dummy_input']
    common_entity = batch['common_entity']
    multi_num = 4
    final_result=torch.zeros([common_entity,4]) if common_entity > 0  else torch.zeros([1,4])
    if common_entity > 0:
        for entity_number in range(common_entity):
            for story_number in range(multi_num):
                entity_input_pair=torch.cat(\
                                        (codah_input[story_number][entity_number]['input_ids'].long().to(device),\
                                         codah_input[story_number][entity_number]['input_ids'].long().to(device)),dim=0)
                entity_mask_pair=torch.cat(\
                                        (codah_input[story_number][entity_number]['attention_mask'].long().to(device),\
                                         codah_input[story_number][entity_number]['attention_mask'].long().to(device)),dim=0)           
                entity_timestep_pair=torch.cat(\
                                        (codah_input[story_number][entity_number]['timestamp_id'].long().to(device),\
                                         codah_input[story_number][entity_number]['timestamp_id'].long().to(device)),dim=0) 

                entity_precondition_pair = torch.zeros([len(entity_input_pair),20]).long().to(device)
                entity_effect_pair = torch.zeros([len(entity_input_pair), 20]).long().to(device)
                entity_conflict_pair = torch.zeros([2,len(codah_input[story_number][entity_number]['input_ids'])]).long().to(device)
                entity_label = torch.tensor([0]).long().to(device)
                segment_ids = None
                with torch.no_grad():
                    entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                                     ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                                    conflict=entity_conflict_pair,label=entity_label,joint_label=0)
                if sentence_tag:
                    final_result[entity_number][story_number]=-torch.sum(entity_out['out_conflict'][0], dim=(0))
                else:
                    final_result[entity_number][story_number]=entity_out['true_out_story'][0]
    else:
        for story_number in range(multi_num):
            entity_input_pair=torch.cat(\
                                    (dummy_input[story_number]['input_ids'].long().to(device),\
                                     dummy_input[story_number]['input_ids'].long().to(device)),dim=0)
            entity_mask_pair=torch.cat(\
                                    (dummy_input[story_number]['attention_mask'].long().to(device),\
                                     dummy_input[story_number]['attention_mask'].long().to(device)),dim=0)           
            entity_timestep_pair=torch.cat(\
                                    (dummy_input[story_number]['timestamp_id'].long().to(device),\
                                     dummy_input[story_number]['timestamp_id'].long().to(device)),dim=0) 

            entity_precondition_pair = torch.zeros([len(entity_input_pair),20]).long().to(device)
            entity_effect_pair = torch.zeros([len(entity_input_pair), 20]).long().to(device)
            entity_conflict_pair = torch.zeros([2,len(dummy_input[story_number]['input_ids'])]).long().to(device)
            entity_label = torch.tensor([0]).long().to(device)
            segment_ids = None
            with torch.no_grad():
                entity_out = tslm(input_ids=entity_input_pair,attention_mask=entity_mask_pair,timestep_type_ids=entity_timestep_pair\
                                 ,token_type_ids=segment_ids,prec_label=entity_precondition_pair,effect_label=entity_effect_pair,\
                                conflict=entity_conflict_pair,label=entity_label,joint_label=0)
            if sentence_tag:
                final_result[0][story_number]=-torch.sum(entity_out['out_conflict'][0], dim=(0))
            else:
                final_result[0][story_number]=entity_out['true_out_story'][0]
            
    softmax = nn.Softmax(dim=1)
    final_result = softmax(final_result)
    label_pred= np.argmax(np.mean(final_result.cpu().numpy(),axis=0))  
    label_pred = np.array([[label_pred]])
    return label_pred,final_result
            
