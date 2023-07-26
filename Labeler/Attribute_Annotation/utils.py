import numpy as np
import json_lines



def read_jsonl(path):
    temp_data = []
    with open(path, 'rb') as f: 
        for item in json_lines.reader(f):
            temp_data.append(item)
    return temp_data




def activateStoryId(tiered_dataset,attribute_id,activate):
    for index,sample in enumerate(tiered_dataset):
        for story_idx,story in enumerate(sample['stories']):
            for entity in story['entities']:
                entity_name = entity['entity']
                prec = entity['preconditions']
                effe = entity['effects']
                for sen_idx,sentence in enumerate(story['sentences']):
                    if prec[sen_idx][attribute_id] == activate:
                        print(index)
                    if effe[sen_idx][attribute_id] == activate:
                        print(index)
                    
def getStoryInformation(samplestory):
    for story_idx,story in enumerate(samplestory['stories']):
        print(story_idx)
        for sen in story['sentences']:
            print(sen)
        print([_['entity'] for _ in story['entities']])

def getActiveAttribute(samplestory,attribute_id):
    attribute_value = ['inactive','active']
    for story_idx,story in enumerate(samplestory['stories']):
        print(story_idx)
        for entity in story['entities']:
            entity_name = entity['entity']
            prec = entity['preconditions']
            effe = entity['effects']
            for sen_idx,sentence in enumerate(story['sentences']):
                if prec[sen_idx][attribute_id]!=0:
                    print(sentence)
                    print("Prec of "+entity_name+" is "+str(attribute_value[int(prec[sen_idx][attribute_id])-1]))
                if effe[sen_idx][attribute_id]!=0:
                    print(sentence)
                    print("Effe of "+entity_name+" is "+str(attribute_value[int(effe[sen_idx][attribute_id])-1]))
                    
                    
def getEntityAttribute(sample_story,entity_name):
    for entity in sample_story['entities']:
        if entity['entity'] == entity_name:
            return entity['preconditions'],entity['effects']
        else:
            continue
            
    return 0,0
        
        
def getAccuracyRecall(store_list,length):
    store_list = np.array([float(item) for item in store_list])
#     store_list[:]+=0.01
    templist=[0]*length*2
    for i in range(length):
        if store_list[i*3+1] == 0 and store_list[i*3+2] == 0:
            templist[i*2] = 0
        else:
            templist[i*2]=store_list[i*3+2]/store_list[i*3+1]
        if store_list[i*3] ==0:
            templist[i*2+1] = 0
        else:
            templist[i*2+1]=store_list[i*3+2]/store_list[i*3]
        
    return templist

def ComputeF1(input_list,length):
    f1=0
    precision = 0
    recall = 0
    for i in range(length):
        precision += input_list[2*i]
        recall += input_list[2*i+1]
    precision = precision/length
    recall = recall/length
        
    return 2 * (precision * recall) / (precision + recall)

def computeCos(vec1,vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_wordemb(output):
    input_ids = output['input_ids'].cpu().numpy()
    token_embeddings = output['token_embeddings'].cpu().numpy()
    word_index = np.where(input_ids==10)[0][-1]
    word_span = np.arange(word_index+1,len(input_ids)-2)
    word_emb = np.zeros(len(token_embeddings[0]))
#     print(word_span)
    for i in (word_span):
        word_emb = word_emb + token_embeddings[i]

    word_emb = word_emb / len(word_span)
    
    return word_emb


def by_value(t):
    return -t['Cos_similarity']




