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


def add_input_feature_joint_dummy_cskg(dataset,tokenizer,maxStoryLength,max_story_length):
    # max_story_length =5
    print("Add input feature joint")
    precondition=[[0]*20 for _ in range(max_story_length)]
    effect=[[0]*20 for _ in range(max_story_length)]
    conflict=[0]*3
    temp_max=0
    for sample_data in tqdm(dataset):
        cskg_input_1=[]
        for entity_index,entity in enumerate(sample_data['entity_1']):
            all_input_ids = np.zeros((max_story_length, maxStoryLength))
            all_input_mask = np.zeros((max_story_length, maxStoryLength))
            all_timestep = np.zeros((max_story_length, maxStoryLength))
            question="Where is "+str(entity)+"?! </s> "
            story=""
            for idx,sentence in enumerate(sample_data['story1']):
                story=story+sentence+" </s> " if idx< len(sample_data['story1'])-1 else story+sentence
            story=[question+story]*len(sample_data['story1'])
            inputs=tokenizer(story)
            f_out=[]
            for time in range(len(sample_data['story1'])):
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
                temp_max = max(len(input_ids[index]),temp_max)
                paddingLength=maxStoryLength-len(input_ids[index])
                all_input_ids[index]=input_ids[index]+[0]*paddingLength
                all_input_mask[index]=attention_mask[index]+[0]*paddingLength
                all_timestep[index]=f_out[index]+[0]*paddingLength
            cskg_input_1.append({'input_ids':all_input_ids.tolist(),
                                'attention_mask':all_input_mask.tolist(),
                                'timestamp_id':all_timestep.tolist(),
                                'precondition':precondition,
                                'effect':effect,
                                'conflict':conflict})
# dummy input 1
        all_input_ids = np.zeros((max_story_length, maxStoryLength))
        all_input_mask = np.zeros((max_story_length, maxStoryLength))
        all_timestep = np.zeros((max_story_length, maxStoryLength))        
        question="Where is _?! </s> "
        story=""
        for idx,sentence in enumerate(sample_data['story1']):
            story=story+sentence+" </s> " if idx< len(sample_data['story1'])-1 else story+sentence
        story=[question+story]*len(sample_data['story1'])
        inputs=tokenizer(story)
        f_out=[]
        for time in range(len(sample_data['story1'])):
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
            all_input_ids[index]=input_ids[index]+[0]*paddingLength
            all_input_mask[index]=attention_mask[index]+[0]*paddingLength
            all_timestep[index]=f_out[index]+[0]*paddingLength
        sample_data['cskg_dummy_1']={'input_ids':all_input_ids.tolist(),
                            'attention_mask':all_input_mask.tolist(),
                            'timestamp_id':all_timestep.tolist(),
                            'precondition':precondition,
                            'effect':effect,
                            'conflict':conflict}
            
            
        cskg_input_2=[]
        for entity_index,entity in enumerate(sample_data['entity_2']):
            all_input_ids = np.zeros((max_story_length, maxStoryLength))
            all_input_mask = np.zeros((max_story_length, maxStoryLength))
            all_timestep = np.zeros((max_story_length, maxStoryLength))  
            question="Where is "+str(entity)+"?! </s> "
            story=""
            for idx,sentence in enumerate(sample_data['story2']):
                story=story+sentence+" </s> " if idx< len(sample_data['story2'])-1 else story+sentence
            story=[question+story]*len(sample_data['story2'])
            inputs=tokenizer(story)
            f_out=[]
            for time in range(len(sample_data['story2'])):
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
                temp_max = max(len(input_ids[index]),temp_max)
                paddingLength=maxStoryLength-len(input_ids[index])
                all_input_ids[index]=input_ids[index]+[0]*paddingLength
                all_input_mask[index]=attention_mask[index]+[0]*paddingLength
                all_timestep[index]=f_out[index]+[0]*paddingLength
            cskg_input_2.append({'input_ids':all_input_ids.tolist(),
                                'attention_mask':all_input_mask.tolist(),
                                'timestamp_id':all_timestep.tolist(),
                                'precondition':precondition,
                                'effect':effect,
                                'conflict':conflict})
# dummy input 2
        all_input_ids = np.zeros((max_story_length, maxStoryLength))
        all_input_mask = np.zeros((max_story_length, maxStoryLength))
        all_timestep = np.zeros((max_story_length, maxStoryLength))            
        question="Where is _?! </s> "
        story=""
        for idx,sentence in enumerate(sample_data['story2']):
            story=story+sentence+" </s> " if idx< len(sample_data['story2'])-1 else story+sentence
        story=[question+story]*len(sample_data['story2'])
        inputs=tokenizer(story)
        f_out=[]
        for time in range(len(sample_data['story2'])):
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
            all_input_ids[index]=input_ids[index]+[0]*paddingLength
            all_input_mask[index]=attention_mask[index]+[0]*paddingLength
            all_timestep[index]=f_out[index]+[0]*paddingLength
        sample_data['cskg_dummy_2']={'input_ids':all_input_ids.tolist(),
                            'attention_mask':all_input_mask.tolist(),
                            'timestamp_id':all_timestep.tolist(),
                            'precondition':precondition,
                            'effect':effect,
                            'conflict':conflict}
        sample_data['cskg_input_1']=cskg_input_1
        sample_data['cskg_input_2']=cskg_input_2
    print(temp_max)
    return dataset


def transform_cskg(sample):
    sample = np.array(sample)
    attribute_list=[list(sample[:,i].astype(int)) for i in range(3)]
        
    return attribute_list




def add_input_feature_joint_dummy_roc(dataset,tokenizer,maxStoryLength,max_story_length):
    # max_story_length =5
    print("Add input feature joint")
    precondition=[[0]*20 for _ in range(max_story_length)]
    effect=[[0]*20 for _ in range(max_story_length)]
    conflict=[0]*10
    temp_max=0
    for sample_data in tqdm(dataset):
        cskg_input_1=[]
        for entity_index,entity in enumerate(sample_data['entity_1']):
            all_input_ids = np.zeros((max_story_length, maxStoryLength))
            all_input_mask = np.zeros((max_story_length, maxStoryLength))
            all_timestep = np.zeros((max_story_length, maxStoryLength))
            question="Where is "+str(entity)+"?! </s> "
            story=""
            for idx,sentence in enumerate(sample_data['goal_sol_1']):
                story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_1'])-1 else story+sentence
            story=[question+story]*len(sample_data['goal_sol_1'])
            inputs=tokenizer(story)
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
                temp_max = max(len(input_ids[index]),temp_max)
                paddingLength=maxStoryLength-len(input_ids[index])
                all_input_ids[index]=input_ids[index]+[0]*paddingLength
                all_input_mask[index]=attention_mask[index]+[0]*paddingLength
                all_timestep[index]=f_out[index]+[0]*paddingLength
            cskg_input_1.append({'input_ids':all_input_ids.tolist(),
                                'attention_mask':all_input_mask.tolist(),
                                'timestamp_id':all_timestep.tolist(),
                                'precondition':precondition,
                                'effect':effect,
                                'conflict':conflict})
# dummy input 1
        all_input_ids = np.zeros((max_story_length, maxStoryLength))
        all_input_mask = np.zeros((max_story_length, maxStoryLength))
        all_timestep = np.zeros((max_story_length, maxStoryLength))        
        question="Where is _?! </s> "
        story=""
        for idx,sentence in enumerate(sample_data['goal_sol_1']):
            story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_1'])-1 else story+sentence
        story=[question+story]*len(sample_data['goal_sol_1'])
        inputs=tokenizer(story)
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
            all_input_ids[index]=input_ids[index]+[0]*paddingLength
            all_input_mask[index]=attention_mask[index]+[0]*paddingLength
            all_timestep[index]=f_out[index]+[0]*paddingLength
        sample_data['cskg_dummy_1']={'input_ids':all_input_ids.tolist(),
                            'attention_mask':all_input_mask.tolist(),
                            'timestamp_id':all_timestep.tolist(),
                            'precondition':precondition,
                            'effect':effect,
                            'conflict':conflict}
            
            
        cskg_input_2=[]
        for entity_index,entity in enumerate(sample_data['entity_2']):
            all_input_ids = np.zeros((max_story_length, maxStoryLength))
            all_input_mask = np.zeros((max_story_length, maxStoryLength))
            all_timestep = np.zeros((max_story_length, maxStoryLength))  
            question="Where is "+str(entity)+"?! </s> "
            story=""
            for idx,sentence in enumerate(sample_data['goal_sol_2']):
                story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_2'])-1 else story+sentence
            story=[question+story]*len(sample_data['goal_sol_2'])
            inputs=tokenizer(story)
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
                temp_max = max(len(input_ids[index]),temp_max)
                paddingLength=maxStoryLength-len(input_ids[index])
                all_input_ids[index]=input_ids[index]+[0]*paddingLength
                all_input_mask[index]=attention_mask[index]+[0]*paddingLength
                all_timestep[index]=f_out[index]+[0]*paddingLength
            cskg_input_2.append({'input_ids':all_input_ids.tolist(),
                                'attention_mask':all_input_mask.tolist(),
                                'timestamp_id':all_timestep.tolist(),
                                'precondition':precondition,
                                'effect':effect,
                                'conflict':conflict})
# dummy input 2
        all_input_ids = np.zeros((max_story_length, maxStoryLength))
        all_input_mask = np.zeros((max_story_length, maxStoryLength))
        all_timestep = np.zeros((max_story_length, maxStoryLength))            
        question="Where is _?! </s> "
        story=""
        for idx,sentence in enumerate(sample_data['goal_sol_2']):
            story=story+sentence+" </s> " if idx< len(sample_data['goal_sol_2'])-1 else story+sentence
        story=[question+story]*len(sample_data['goal_sol_2'])
        inputs=tokenizer(story)
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
            all_input_ids[index]=input_ids[index]+[0]*paddingLength
            all_input_mask[index]=attention_mask[index]+[0]*paddingLength
            all_timestep[index]=f_out[index]+[0]*paddingLength
        sample_data['cskg_dummy_2']={'input_ids':all_input_ids.tolist(),
                            'attention_mask':all_input_mask.tolist(),
                            'timestamp_id':all_timestep.tolist(),
                            'precondition':precondition,
                            'effect':effect,
                            'conflict':conflict}
        sample_data['cskg_input_1']=cskg_input_1
        sample_data['cskg_input_2']=cskg_input_2
    print(temp_max)
    return dataset


def transform_roc(sample):
    sample = np.array(sample)
    attribute_list=[list(sample[:,i].astype(int)) for i in range(5)]
        
    return attribute_list