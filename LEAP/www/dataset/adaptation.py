import torch
from torch.utils.data import TensorDataset
from www.utils import get_sublist
import spacy
from www.dataset.ann import human_atts, att_to_idx, att_types
import progressbar
from time import sleep
from copy import deepcopy
import numpy as np
from tqdm import tqdm


def checkdistribution(tiered_dataset,tiered_tensor_dataset):
    for p in tiered_dataset:
        dataset=tiered_dataset[p]
        print("{}dataset:{}:".format(p,tiered_tensor_dataset[p][1][0].shape))
        entity = [len(story['entities']) for ex_2s in dataset for story in ex_2s['stories']]
        count={}
        for item in entity:
            if item not in count.keys():
                count[item]=1
            else:
                count[item]+=1
        print(sorted(count.items(), key = lambda kv:(kv[0], kv[1]))) 
        
def filter_dataset(dataset,num_limited):
    for p in dataset:
        newdataset=[]
        for item in dataset[p]:
            if len(item['stories'][0]['entities'])<=num_limited and len(item['stories'][1]['entities'])<=num_limited:
                newdataset.append(item)
        print("for {} dataset, number change from {} to {}".format(p,len(dataset[p]),len(newdataset)))
        dataset[p]=newdataset
    return dataset

def getMaxStoryLength(dataset,tokenizer):
    maxStoryLength=0
    lengthTotal=[]
    for p in dataset:
        for i, ex_2s in enumerate(dataset[p]):
            for s_idx, ex_1s in enumerate(ex_2s['stories']):
                for ent_idx, ex in enumerate(ex_1s['entities']):
                    question="Where is "+str(ex['entity']+"?! </s> ")
                    story=""
                    for idx,sentence in enumerate(ex['sentences']):
                        story=story+sentence+" </s> " if idx< len(ex['sentences'])-1 else story+sentence
                    qaStories=[question+story]*5
                    inputs=tokenizer(qaStories)
                    lengthTotal.append(len(inputs['input_ids'][0]))
                    maxStoryLength=max(maxStoryLength,len(inputs['input_ids'][0]))
    return maxStoryLength,lengthTotal

def add_bert_features_tiered_modify(dataset,tokenizer,maxStoryLength,add_segment_ids=False):
    nlp = spacy.load("en_core_web_sm")
    max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
    for p in dataset:
        bar_size = len(dataset[p])
        bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar_idx = 0
        bar.start()
        for i, ex_2s in enumerate(dataset[p]):
            for s_idx, ex_1s in enumerate(ex_2s['stories']):
                for ent_idx, ex in enumerate(ex_1s['entities']):
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    question="Where is "+str(ex['entity']+"?! </s> ")
                    story=""
                    for idx,sentence in enumerate(ex['sentences']):
                        story=story+sentence+" </s> " if idx< len(ex['sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(ex['sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(ex['sentences'])):
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
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["timestep_type_ids"]=all_timestep
            bar_idx += 1
            bar.update(bar_idx)
        bar.finish()
    return dataset


def add_bert_features_tiered_joint(dataset,tokenizer,maxStoryLength,add_segment_ids=False):
    nlp = spacy.load("en_core_web_sm")
    max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
    for p in dataset:
        bar_size = len(dataset[p])
        bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar_idx = 0
        bar.start()
        for i, ex_2s in enumerate(dataset[p]):
            for s_idx, ex_1s in enumerate(ex_2s['stories']):
                for ent_idx, ex in enumerate(ex_1s['entities']):
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    opposite_story=ex_2s['stories'][1-s_idx]
                    opposite_story_sen=""
                    for sen in opposite_story['sentences']:
                        opposite_story_sen+=sen+" "
                    opposite_story_sen=opposite_story_sen[:-1]
                    question="Where is "+str(ex['entity']+"?! and the opposite story is "+opposite_story_sen+" </s> ")
                    story=""
                    for idx,sentence in enumerate(ex['sentences']):
                        story=story+sentence+" </s> " if idx< len(ex['sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(ex['sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(ex['sentences'])):
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
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["timestep_type_ids"]=all_timestep
            bar_idx += 1
            bar.update(bar_idx)
        bar.finish()
    return dataset


def get_tensor_dataset_tiered_modify(dataset,max_sentences,maxStoryLength,add_segment_ids=False):
    max_entities = max([len(story['entities']) for ex_2s in dataset for story in ex_2s['stories']])
    num_attributes = len(dataset[0]['stories'][0]['entities'][0]['preconditions'][0])

    all_input_ids = torch.tensor([[[[story['entities'][e]['input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_lengths = torch.tensor([[[len(story['sentences']) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    num_entities = torch.tensor([[len(story['entities']) for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    all_input_mask = torch.tensor([[[[story['entities'][e]['attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_attributes = torch.tensor([[[[story['entities'][e]['attributes'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_preconditions = torch.tensor([[[[story['entities'][e]['preconditions'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_effects = torch.tensor([[[[story['entities'][e]['effects'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_spans = torch.tensor([[[story['entities'][e]['conflict_span_onehot'] if e < len(story['entities']) else np.zeros((max_sentences)) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_label_ids = torch.tensor([ex['label'] for ex in dataset], dtype=torch.long)
    
    all_timestep_type_ids=torch.tensor([[[[story['entities'][e]['timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_plausbile = torch.tensor([[[story['entities'][e]['plausible'] if e < len(story['entities']) else 0 
                            for e in range(max_entities)] 
                            for story in ex_2s['stories']] 
                            for ex_2s in dataset], dtype=torch.long)
    
    all_common_entity_number=torch.tensor([ex['common_entity'] for ex in dataset], dtype=torch.long)
    
    tensor_dataset = TensorDataset(all_input_ids, all_lengths, num_entities, all_input_mask, all_attributes, all_preconditions, all_effects, all_spans, all_label_ids,all_timestep_type_ids,all_plausbile,all_common_entity_number)
    return tensor_dataset



def add_bert_features_tiered_joint_cut(dataset,tokenizer,maxStoryLength,add_segment_ids=False):
    nlp = spacy.load("en_core_web_sm")
    max_story_length = max([len(ex['sentences']) for ex_2s in dataset for ex in ex_2s['stories']])
    print(max_story_length)
    bar_size = len(dataset)
    bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar_idx = 0
    bar.start()
    for i, ex_2s in enumerate(dataset):
        for s_idx, ex_1s in enumerate(ex_2s['stories']):
            for ent_idx, ex in enumerate(ex_1s['entities']):
                all_input_ids = np.zeros((max_story_length, maxStoryLength))
                all_input_mask = np.zeros((max_story_length, maxStoryLength))
                all_timestep = np.zeros((max_story_length, maxStoryLength))
                opposite_story=ex_2s['stories'][1-s_idx]
                opposite_story_sen=""
                for sen in opposite_story['sentences']:
                    opposite_story_sen+=sen+" "
                opposite_story_sen=opposite_story_sen[:-1]
                question="Where is "+str(ex['entity']+"?! and the opposite story is "+opposite_story_sen+" </s> ")
                story=""
                for idx,sentence in enumerate(ex['sentences']):
                    story=story+sentence+" </s> " if idx< len(ex['sentences'])-1 else story+sentence
                qaStories=[question+story]*len(ex['sentences'])
                inputs=tokenizer(qaStories)
                f_out=[]
                for time in range(len(ex['sentences'])):
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
                dataset[i]['stories'][s_idx]['entities'][ent_idx]["input_ids"]=all_input_ids
                dataset[i]['stories'][s_idx]['entities'][ent_idx]["attention_mask"]=all_input_mask
                dataset[i]['stories'][s_idx]['entities'][ent_idx]["timestep_type_ids"]=all_timestep
        bar_idx += 1
        bar.update(bar_idx)
    bar.finish()
    return dataset



def ReOrderDataset(dataset):
    for p in dataset.keys():
        for sample in tqdm(dataset[p]):
            list1 = [_['entity'] for _ in sample['stories'][0]['entities']]
            list2 = [_['entity'] for _ in sample['stories'][1]['entities']]
            entity_set=set(list1)&set(list2)
            entity_list=sorted(list(entity_set))
            sample['common_entity']=len(entity_list)
            entity_list1=[]
            entity_list2=[]
            for entity in entity_list:
                entity_list1.append(sample['stories'][0]['entities'][list1.index(entity)])
                entity_list2.append(sample['stories'][1]['entities'][list2.index(entity)])
            entity_left_1=set(list1)-entity_set
            entity_left_2=set(list2)-entity_set
            for temp_entity in entity_left_1:
                entity_list1.append(sample['stories'][0]['entities'][list1.index(temp_entity)])

            for temp_entity in entity_left_2:
                entity_list2.append(sample['stories'][1]['entities'][list2.index(temp_entity)])
            sample['stories'][0]['entities']=entity_list1
            sample['stories'][1]['entities']=entity_list2
    return dataset




def add_input_feature_new_joint(dataset,tokenizer):
    print("Add input feature joint")
    for sample_data in tqdm(dataset):
        roc_input_1=[]
        for entity in sample_data['entity_1']:
            opposite_story=sample_data['goal_sol_2']
            opposite_story_sen=""
            for sen in opposite_story:
                opposite_story_sen+=sen+" "
            opposite_story_sen=opposite_story_sen[:-1]
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
            roc_input_1.append({'input_ids':torch.tensor(input_ids),
                                'attention_mask':torch.tensor(attention_mask),
                                'timestamp_id':torch.tensor(f_out)})
            
        roc_input_2=[]
        for entity in sample_data['entity_2']:
            opposite_story=sample_data['goal_sol_1']
            opposite_story_sen=""
            for sen in opposite_story:
                opposite_story_sen+=sen+" "
            opposite_story_sen=opposite_story_sen[:-1]
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
            roc_input_2.append({'input_ids':torch.tensor(input_ids),
                                'attention_mask':torch.tensor(attention_mask),
                                'timestamp_id':torch.tensor(f_out)})
        sample_data['piqa_input_1']=roc_input_1
        sample_data['piqa_input_2']=roc_input_2
    return dataset




#### MODEL_OPPOSITE
def get_tensor_dataset_tiered_opposite(dataset,max_sentences,maxStoryLength,add_segment_ids=False):
    max_entities = max([len(story['entities']) for ex_2s in dataset for story in ex_2s['stories']])
    num_attributes = len(dataset[0]['stories'][0]['entities'][0]['preconditions'][0])

    all_input_ids = torch.tensor([[[[story['entities'][e]['input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_lengths = torch.tensor([[[len(story['sentences']) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    num_entities = torch.tensor([[len(story['entities']) for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    all_input_mask = torch.tensor([[[[story['entities'][e]['attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_attributes = torch.tensor([[[[story['entities'][e]['attributes'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_preconditions = torch.tensor([[[[story['entities'][e]['preconditions'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_effects = torch.tensor([[[[story['entities'][e]['effects'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_spans = torch.tensor([[[story['entities'][e]['conflict_span_onehot'] if e < len(story['entities']) else np.zeros((max_sentences)) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_label_ids = torch.tensor([ex['label'] for ex in dataset], dtype=torch.long)
    
    all_timestep_type_ids=torch.tensor([[[[story['entities'][e]['timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_plausbile = torch.tensor([[[story['entities'][e]['plausible'] if e < len(story['entities']) else 0 
                            for e in range(max_entities)] 
                            for story in ex_2s['stories']] 
                            for ex_2s in dataset], dtype=torch.long)
    
    all_opposite_input_ids = torch.tensor([[[[story['entities'][e]['opposite_input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_opposite_input_mask = torch.tensor([[[[story['entities'][e]['opposite_attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_opposite_timestep_type_ids=torch.tensor([[[[story['entities'][e]['opposite_timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    
    tensor_dataset = TensorDataset(all_input_ids, all_lengths, num_entities, all_input_mask, all_attributes, all_preconditions, all_effects, all_spans, all_label_ids,all_timestep_type_ids,all_plausbile,all_opposite_input_ids,all_opposite_input_mask,all_opposite_timestep_type_ids)
    return tensor_dataset


def add_bert_features_tiered_opposite(dataset,tokenizer,maxStoryLength,add_segment_ids=False):
    nlp = spacy.load("en_core_web_sm")
    max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
    for p in dataset:
        bar_size = len(dataset[p])
        bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar_idx = 0
        bar.start()
        for i, ex_2s in enumerate(dataset[p]):
            for s_idx, ex_1s in enumerate(ex_2s['stories']):
                for ent_idx, ex in enumerate(ex_1s['entities']):
                    # for entity input
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    question="Where is "+str(ex['entity'])+"?! </s> "
                    story=""
                    for idx,sentence in enumerate(ex['sentences']):
                        story=story+sentence+" </s> " if idx< len(ex['sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(ex['sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(ex['sentences'])):
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
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["timestep_type_ids"]=all_timestep
                    
            # for opposite story
                    op_story = ex_2s['stories'][1-s_idx]
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    question="The opposite story is: </s> "
                    story=""
                    for idx,sentence in enumerate(op_story['sentences']):
                        story=story+sentence+" </s> " if idx< len(op_story['sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(op_story['sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(op_story['sentences'])):
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
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["opposite_input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["opposite_attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["opposite_timestep_type_ids"]=all_timestep
            bar_idx += 1
            bar.update(bar_idx)
        bar.finish()
    return dataset


### MODEL_OPPOSITE_WORDNET


def add_bert_features_tiered_opposite_wordnet(dataset,tokenizer,maxStoryLength,add_segment_ids=False):
    nlp = spacy.load("en_core_web_sm")
    max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
    for p in dataset:
        bar_size = len(dataset[p])
        bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar_idx = 0
        bar.start()
        for i, ex_2s in enumerate(dataset[p]):
            for s_idx, ex_1s in enumerate(ex_2s['stories']):
                for ent_idx, ex in enumerate(ex_1s['entities']):
                    # for entity input
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    question="Where is "+str(ex['entity'])+"?! </s> "
                    story=""
                    for idx,sentence in enumerate(ex['sentences']):
                        story=story+sentence+" </s> " if idx< len(ex['sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(ex['sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(ex['sentences'])):
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
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["timestep_type_ids"]=all_timestep

                   # for entity input
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    question="Where is "+str(ex['abs_class'])+"?! </s> "
                    story=""
                    for idx,sentence in enumerate(ex['abs_sentences']):
                        story=story+sentence+" </s> " if idx< len(ex['abs_sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(ex['abs_sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(ex['abs_sentences'])):
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
#                         if len(input_ids[index]) <= maxStoryLength:
#                             print(maxStoryLength)
                        paddingLength=maxStoryLength-len(input_ids[index])
                        all_input_ids[index]=input_ids[index]+[0]*paddingLength
                        all_input_mask[index]=attention_mask[index]+[0]*paddingLength
                        all_timestep[index]=f_out[index]+[0]*paddingLength
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["wordnet_input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["wordnet_attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["wordnet_timestep_type_ids"]=all_timestep
                    
            # for opposite story
                    op_story = ex_2s['stories'][1-s_idx]
                    all_input_ids = np.zeros((max_story_length, maxStoryLength))
                    all_input_mask = np.zeros((max_story_length, maxStoryLength))
                    all_timestep = np.zeros((max_story_length, maxStoryLength))
                    question="The opposite story is: </s> "
                    story=""
                    for idx,sentence in enumerate(op_story['sentences']):
                        story=story+sentence+" </s> " if idx< len(op_story['sentences'])-1 else story+sentence
                    qaStories=[question+story]*len(op_story['sentences'])
                    inputs=tokenizer(qaStories)
                    f_out=[]
                    for time in range(len(op_story['sentences'])):
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
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["opposite_input_ids"]=all_input_ids
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["opposite_attention_mask"]=all_input_mask
                    dataset[p][i]['stories'][s_idx]['entities'][ent_idx]["opposite_timestep_type_ids"]=all_timestep
            bar_idx += 1
            bar.update(bar_idx)
        bar.finish()
    return dataset



def get_tensor_dataset_tiered_opposite_wordnet(dataset,max_sentences,maxStoryLength,add_segment_ids=False):
    max_entities = max([len(story['entities']) for ex_2s in dataset for story in ex_2s['stories']])
    num_attributes = len(dataset[0]['stories'][0]['entities'][0]['preconditions'][0])

    all_input_ids = torch.tensor([[[[story['entities'][e]['input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_lengths = torch.tensor([[[len(story['sentences']) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    num_entities = torch.tensor([[len(story['entities']) for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    all_input_mask = torch.tensor([[[[story['entities'][e]['attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_attributes = torch.tensor([[[[story['entities'][e]['attributes'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_preconditions = torch.tensor([[[[story['entities'][e]['preconditions'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_effects = torch.tensor([[[[story['entities'][e]['effects'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_spans = torch.tensor([[[story['entities'][e]['conflict_span_onehot'] if e < len(story['entities']) else np.zeros((max_sentences)) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_label_ids = torch.tensor([ex['label'] for ex in dataset], dtype=torch.long)
    
    all_timestep_type_ids=torch.tensor([[[[story['entities'][e]['timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_plausbile = torch.tensor([[[story['entities'][e]['plausible'] if e < len(story['entities']) else 0 
                            for e in range(max_entities)] 
                            for story in ex_2s['stories']] 
                            for ex_2s in dataset], dtype=torch.long)
    
    all_opposite_input_ids = torch.tensor([[[[story['entities'][e]['opposite_input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_opposite_input_mask = torch.tensor([[[[story['entities'][e]['opposite_attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_opposite_timestep_type_ids=torch.tensor([[[[story['entities'][e]['opposite_timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    
    
    all_wordnet_input_ids = torch.tensor([[[[story['entities'][e]['wordnet_input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_wordnet_input_mask = torch.tensor([[[[story['entities'][e]['wordnet_attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_wordnet_timestep_type_ids=torch.tensor([[[[story['entities'][e]['wordnet_timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    
    tensor_dataset = TensorDataset(all_input_ids, all_lengths, num_entities, all_input_mask, all_attributes, all_preconditions, all_effects, all_spans, all_label_ids,all_timestep_type_ids,all_plausbile,all_opposite_input_ids,all_opposite_input_mask,all_opposite_timestep_type_ids,all_wordnet_input_ids,all_wordnet_input_mask,all_wordnet_timestep_type_ids)
    return tensor_dataset


# Method for joint model dummy

def add_bert_features_tiered_dummy(dataset,tokenizer,maxStoryLength,add_segment_ids=False):
    nlp = spacy.load("en_core_web_sm")
    max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
    for p in dataset:
        for i, ex_2s in enumerate(dataset[p]):
            for s_idx, ex_1s in enumerate(ex_2s['stories']):
                all_input_ids = np.zeros((max_story_length, maxStoryLength))
                all_input_mask = np.zeros((max_story_length, maxStoryLength))
                all_timestep = np.zeros((max_story_length, maxStoryLength))
                question="Where is _?! </s> "
                story=""
                for idx,sentence in enumerate(ex_1s['sentences']):
                    story=story+sentence+" </s> " if idx< len(ex_1s['sentences'])-1 else story+sentence
                qaStories=[question+story]*len(ex_1s['sentences'])
                inputs=tokenizer(qaStories)
                f_out=[]
                for time in range(len(ex_1s['sentences'])):
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
                dataset[p][i]['stories'][s_idx]["dummy_input_ids"]=all_input_ids
                dataset[p][i]['stories'][s_idx]["dummy_attention_mask"]=all_input_mask
                dataset[p][i]['stories'][s_idx]["dummy_timestep_type_ids"]=all_timestep
    return dataset


from torch.utils.data import TensorDataset
def get_tensor_dataset_tiered_dummy(dataset,max_sentences,maxStoryLength,add_segment_ids=False):
    max_entities = max([len(story['entities']) for ex_2s in dataset for story in ex_2s['stories']])
    num_attributes = len(dataset[0]['stories'][0]['entities'][0]['preconditions'][0])

    all_input_ids = torch.tensor([[[[story['entities'][e]['input_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_lengths = torch.tensor([[[len(story['sentences']) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    num_entities = torch.tensor([[len(story['entities']) for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
    all_input_mask = torch.tensor([[[[story['entities'][e]['attention_mask'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_attributes = torch.tensor([[[[story['entities'][e]['attributes'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_preconditions = torch.tensor([[[[story['entities'][e]['preconditions'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_effects = torch.tensor([[[[story['entities'][e]['effects'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_spans = torch.tensor([[[story['entities'][e]['conflict_span_onehot'] if e < len(story['entities']) else np.zeros((max_sentences)) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
    all_label_ids = torch.tensor([ex['label'] for ex in dataset], dtype=torch.long)
    
    all_timestep_type_ids=torch.tensor([[[[story['entities'][e]['timestep_type_ids'][s] if e < len(story['entities']) else np.zeros((maxStoryLength)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    all_plausbile = torch.tensor([[[story['entities'][e]['plausible'] if e < len(story['entities']) else 0 
                            for e in range(max_entities)] 
                            for story in ex_2s['stories']] 
                            for ex_2s in dataset], dtype=torch.long)
    
    all_common_entity_number=torch.tensor([ex['common_entity'] for ex in dataset], dtype=torch.long)
    dummy_input_ids = torch.tensor([[story['dummy_input_ids'] for story in ex_2s['stories']] for ex_2s in dataset])
    dummy_attention_mask = torch.tensor([[story['dummy_attention_mask'] for story in ex_2s['stories']] for ex_2s in dataset])
    dummy_timestep_type_ids = torch.tensor([[story['dummy_timestep_type_ids'] for story in ex_2s['stories']] for ex_2s in dataset])
    tensor_dataset = TensorDataset(all_input_ids, all_lengths, num_entities, all_input_mask, all_attributes, all_preconditions, all_effects, all_spans, all_label_ids,all_timestep_type_ids,all_plausbile,all_common_entity_number,dummy_input_ids,\
                                  dummy_attention_mask,dummy_timestep_type_ids)
    return tensor_dataset