import os
import openai
import string
from utils import *
from www.dataset.ann import *
from tqdm import tqdm
import time
import json
from www.dataset.ann import *
from collections import Counter
from www.dataset.prepro import get_tiered_data, balance_labels
from www.dataset.adaptation import ReOrderDataset,add_bert_features_tiered_modify,add_bert_features_tiered_dummy,get_tensor_dataset_tiered_dummy
from collections import Counter
from transformers import RobertaTokenizerFast
robertatokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
from sentence_transformers import SentenceTransformer
SentenceTransformer = SentenceTransformer('sentence-transformers/sentence-t5-base')





def GetSentencePrompt(story):
    sentences_prompt="def main():\n\t# Init"
    sentences = story['sentences']
    for sentence in sentences:
        sentences_prompt = sentences_prompt + "\n\t# {sentence}".format(sentence = sentence)
    return  sentences_prompt

def getPromptingAll(story,attribute_idx,state,input_text=False):
    ###
    sentence_prompt = GetSentencePrompt(story)
    sentence_length = len(story['sentences'])
    sentence_prompt += '\n\t#Entity'
    entity_list = [_['entity'] for _ in story['entities']]
    for entity in entity_list:
        entity_prompy = '{entity} = Node()'.format(entity=entity)
        sentence_prompt += '\n\t'+entity_prompy
    
    if input_text:
        return sentence_prompt + '\n\tdef'

    for idx,sentence in enumerate(story['sentences']):
        pass_flag = True
        sentence_prompt = sentence_prompt + "\n\tdef {sentence}():".format(sentence = sentence.strip(string.punctuation).replace(" ","_"))
        for entity_idx,entity in enumerate(story['entities']):
            state_result = None
            if entity['entity'] in sentence:
                entity_attribute_list = entity[state][idx]
                entity_attribute = int(entity_attribute_list[attribute_idx])
                if entity_attribute >0:
                    state_result = att_adj[idx_to_att[attribute_idx]][entity_attribute-1]
                    sentence_prompt = sentence_prompt + '\n\t\t {entity}.{attribute}.activate()'\
                    .format(entity=entity['entity'],attribute=idx_to_att[attribute_idx])
                    pass_flag = False
        if pass_flag:
            sentence_prompt+='\n\t\t pass'

    sentence_prompt+='\n#END'
    
    return sentence_prompt

def filter_example(original_list,division,previous_list,tiered_dataset):

    example_list = []
    filter_list = []
    for story_infor in original_list:
        temp_id = story_infor[0]
        story_example_id = int(tiered_dataset['train'][temp_id]['example_id'].split('-')[0])
        if story_example_id in example_list:
            continue
        else:
            example_list.append(story_example_id)
            diverse_flag = True
            for possible_id in filter_list:
                if abs(possible_id-temp_id)<10:
                    diverse_flag = False
            if diverse_flag and temp_id not in previous_list:
                filter_list.append(temp_id)
     
    return filter_list

def FindActivateStory(tiered_dataset,division,attribute,state):
    attribute_id = att_to_idx[attribute]
#     activate_label_list_length = len(att_adj[attribute]) if 'location' in attribute else len(att_adj[attribute])+1

    activate_label_list=[]
    for i in range(len(att_adj[attribute])+1):
        activate_label_list.append({})

    for index,story in enumerate(tiered_dataset[division]):
        for single_story in story['stories']:
            if single_story['plausible'] == True:
                for entity in single_story['entities']:
                    for idx in range(len(single_story['sentences'])):
                        activate_attribute_label = entity[state][idx][attribute_id]
                        if index in activate_label_list[int(activate_attribute_label)].keys():
                            activate_label_list[int(activate_attribute_label)][index] += 1
                        else:
                            activate_label_list[int(activate_attribute_label)][index] = 1
                            
    
    return activate_label_list

def getProProming(attribute,mannual_sample,prompting_numb,tiered_dataset,state,mannual=False):
    attribute_id = att_to_idx[attribute]
    activate_label_list = FindActivateStory(tiered_dataset,'train',attribute,state)[1:]
    
    for list_index in range(len(activate_label_list)):
        activate_label_list[list_index] = sorted(activate_label_list[list_index].items(), key = lambda x:-x[1])  
    
    previous_list = []
    for index in range(len(activate_label_list)):
        activate_label_list[index] = filter_example(activate_label_list[index],'train',previous_list,tiered_dataset)[:prompting_numb-1]+filter_example(activate_label_list[index],'train',previous_list,tiered_dataset)[-1:]
        previous_list = activate_label_list[index]
        
#     print(activate_label_list)
    Prompting = ""
    for sub_list in activate_label_list:
        for i in sub_list:
            label_story = tiered_dataset['train'][i]['stories'][tiered_dataset['train'][i]['label']]
            Prompting = Prompting + getPromptingAll(label_story,attribute_id,state)+'\n\n'

    if mannual:
        Prompting += mannual_sample


    return Prompting


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




def get_attribute_dataset(tiered_dataset,Name_list,attribute_id,state):
    if attribute_id<5:
        attribute_sen={'0':[],'1':[],'2':[]}
        for story in tiered_dataset['train']:
            for single_story in story['stories']:
                for entity in single_story['entities']:
                    if entity['entity'] in Name_list:
                        effect = entity[state]
                        sentences = entity['sentences']
                        for sen_id,sen in enumerate(sentences):
                            #### different names all convert to Someone
                            for name in Name_list:
                                if name in sen:
                                    sen = sen.replace(name,"Someone")
                                    option = effect[sen_id][attribute_id]
                                    attribute_sen[str(int(option))].append(sen+" Focus: Someone.")
                                    break
    elif attribute_id == 5:
        location_dict= {0: 'does not move to a new location', 1: 'disappears', 2: 'is picked up', 3: 'is put down', 4: 'is put on', 5: 'is removed', 6: 'is put into a container', 7: 'is taken out of a container', 8: 'moved somewhere new'}
        location_option_list = [location_dict[key] for key in location_dict.keys()]   
        attribute_sen = {}
        for i in range(9):
            attribute_sen[str(i)]= []
        for story in tiered_dataset['train']:
            for single_story in story['stories']:
                if single_story['plausible']:
                    for entity in single_story['entities']:
                        if entity['entity'] not in Name_list:
                            effect = entity[state]
                            sentences = entity['sentences']
                            for sen_id,sen in enumerate(sentences):
                                #### different names all convert to Someone
                                for name in Name_list:
                                    sen = sen.replace(name,"Someone")
                                if " "+entity['entity'] in sen:
                                    option = effect[sen_id][attribute_id]
                                    attribute_sen[str(int(option))].append(sen+" Focus: "+entity['entity']+".")
    else:
        attribute_sen={'0':[],'1':[],'2':[]}
        for story in tiered_dataset['train']:
            for single_story in story['stories']:
                for entity in single_story['entities']:
                    if entity['entity'] not in Name_list:
                        effect = entity[state]
                        sentences = entity['sentences']
                        for sen_id,sen in enumerate(sentences):
                            #### different names all convert to Someone
                            for name in Name_list:
                                sen = sen.replace(name,"Someone")
                            if " "+entity['entity'] in sen:
                                option = effect[sen_id][attribute_id]
                                attribute_sen[str(int(option))].append(sen+" Focus: "+entity['entity']+".")
                                
    for key,value in attribute_sen.items():
        attribute_sen[key]=list(set(value))
        print(key+": "+str(len(attribute_sen[key])))
    
    for key in attribute_sen.keys():
        distance_list=[]
        for item in tqdm(attribute_sen[key]):
            output = SentenceTransformer.encode(item,output_value = None)
            word_emb = get_wordemb(output)
            distance_list.append({'sentence':item,\
                                  'sentence_embedding':output['sentence_embedding'].cpu().numpy(),\
                                  'word_embeddings':word_emb})
        attribute_sen[key] = distance_list
        
    return attribute_sen


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:                  
        os.makedirs(path)           
        print('New folder created')
    else:
        print("Folder exist")
        
        
        
def getStoryPrompting_CSKG(story,entity_list):

    sentences_prompt="def main():\n\t# Init"

    for sentence in story:
        sentence=sentence[:-1]+'.'
        sentences_prompt = sentences_prompt + "\n\t# {sentence}".format(sentence = sentence)
    sentences_prompt += '\n\t#Entity'
    entity_list = [ _ for _ in entity_list]
    for entity in entity_list:
        entity_prompy = '{entity} = Node()'.format(entity=entity)
        sentences_prompt += '\n\t'+entity_prompy

    sentences_prompt += '\n\tdef'
    
    return sentences_prompt


def getSingleprompt(attribute,sentence_entity,attribute_label,input_text=False):
    sentences_prompt="def main():\n\t# Init"
    sentence = sentence_entity[0]
    entity = sentence_entity[1]
    sentences_prompt +=  "\n\t# {sentence}".format(sentence = sentence)
    sentences_prompt +=  "\n\t# Entity\n\t{entity}= Node()".format(entity = entity.lstrip())
    if input_text:
        return sentences_prompt + '\n\tdef'
    sentences_prompt += "\n\tdef {sentence}():".format(sentence = sentence.strip(string.punctuation).replace(" ","_"))
    sentences_prompt += "\n\t\t {entity}.{attribute} = {attribute_label}".format(entity = entity,attribute = attribute,attribute_label=attribute_label)
    sentences_prompt +='\n#END'
    return sentences_prompt
        
        
        
