import os
import json
import sys
import torch
import random
import numpy as np
import spacy
from tqdm import tqdm
import pickle as pk
import json
import csv
import requests
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import ElementTree
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
nlp = spacy.load('en_core_web_sm')

from spacy.lang.en import English
nlp_sen = English()
config = {"punct_chars": None}
nlp_sen.add_pipe("sentencizer", config=config)

noise_word = np.load('noise_word.npy',allow_pickle = True)

import ntpath
import logging


import argparse
import time
from typing import NamedTuple, List, Optional, Tuple

os.environ['CUDA_PATH']='/usr/local/cuda'
os.environ['LD_LIBRARY_PATH']='$CUDA_PATH/lib64:$LD_LIBRARY_PATH'
ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)


import torch
from torch.utils.data import DataLoader

from esc.utils.definitions_tokenizer import get_tokenizer
from esc.utils.wordnet import synset_from_offset
from esc.utils.wsd import WSDInstance
from esc.esc_dataset import WordNetDataset, OxfordDictionaryDataset
from esc.esc_pl_module import ESCModule
from esc.utils.commons import list_elems_in_dir
from esc.predict import InstancePredictionReport,ScoresReport,PredictionReport,probabilistic_prediction,precision_recall_f1_accuracy_score,predict,process_prediction_result

import ntpath
import logging

### parameter and model
args={}
args['ckpt'] = './ESC_Model/escher_semcor_best.ckpt'
args['dataset_paths'] = 'Test.data.xml'
args['prediction_types'] = 'probabilistic'
args['evaluate'] = False
args['device'] = 0
args['tokens_per_batch'] = 4000
args['output_errors'] = False
args['oxford_test'] = False
wsd_model = ESCModule.load_from_checkpoint(args['ckpt'])
wsd_model.freeze()

tokenizer = get_tokenizer(
    wsd_model.hparams.transformer_model, getattr(wsd_model.hparams, "use_special_tokens", False)
)

prediction_type = args['prediction_types']

dataset_path = args['dataset_paths']
dataset_path = dataset_path.replace(".data.xml", "").replace(".gold.key.txt", "")

def ConstructXml(nlp,sen,entity):
    doc=nlp(sen)
    pos_list=[t.pos_ for t in doc]
    token_list=[t.lemma_ for t in doc]
    word_list = [str(t) for t in doc]

    TargetEntity = [entity]
    special_word = ['others','what','where','when','how','who','everything','call','number']
    for special in special_word:
        if entity in special:
            return 0
    corpus = Element('corpus')
    corpus.attrib = {"lang":"en","source":"Test"}
    text = SubElement(corpus, 'text')
    text.attrib = {"id":"d000"}
    sentence = SubElement(text, 'sentence')
    sentence.attrib = {"id":"d000.s000"}
    for lem,pos,word in zip(token_list,pos_list,word_list):
        temp_pos = 'NOUN'
        if word in TargetEntity:
            if pos == 'VERB' and 'ing' in entity:
                return 0
            
            instance = SubElement(sentence, 'instance')
            instance.attrib = {'id':'d000.s000.t00'+str(TargetEntity.index(word)),'lemma':word.lower(),'pos':temp_pos}
            instance.text = str(word)
        else:
            wf = SubElement(sentence, 'wf')
            wf.attrib = {'lemma':lem.lower(),'pos':temp_pos}
            wf.text = word
    file_handle = open("Test.data.xml","wb")
    tree = ElementTree(corpus)
    tree.write(file_handle)
    file_handle.close()
    
    return 1
    
def checkPhysicalEsc(possible_entity):
    special_word = ['others','what','where','when','how','who','everything','call','number']
    for special in special_word:
        if possible_entity.name() in special:
            return 0
    abstract_word = ['abstraction','abstract_entity','abstract','location','part','piece']
    if len(possible_entity.hypernyms())!=0:
        while True:
            if len(possible_entity.hypernyms())!=0:
                possible_entity=possible_entity.hypernyms()[0]
#                 print(possible_entity.name())
                for abstract in abstract_word:
                    if possible_entity.name().split('.')[0] in abstract:
                        return 0
                if 'physical_entity' in possible_entity.name() or 'physical_object' in possible_entity.name():
                    return 1
            else:
                return 2
    else:
        return 2

def PredictSynsets(dataset_path,tokenizer,args,wsd_model,prediction_type,debug=False):
    dataset = WordNetDataset(
    dataset_path, tokenizer, args['tokens_per_batch'], re_init_on_iter=False, is_test=True
    )

    data_loader = DataLoader(dataset, batch_size=None, num_workers=0)

    prediction_report = predict(
        model=wsd_model,
        data_loader=data_loader,
        device=args['device'],
        prediction_type=prediction_type,
        evaluate=args['evaluate'],
    )
    offset = prediction_report[0][0].predicted_synsets[0]
    if debug:
        print(prediction_report[0][0])
        print(offset)
    return checkPhysicalEsc(wn.of2ss(offset))

def ESCCheck(nlp,sen,entity,dataset_path,tokenizer,args,wsd_model,prediction_type,debug=False):
    result = ConstructXml(nlp,sen,entity)
    if result == 0:
        return 0
    else:
        return PredictSynsets(dataset_path,tokenizer,args,wsd_model,prediction_type,debug)

def CheckAbstract(nlp,sen,entity,dataset_path,tokenizer,args,wsd_model,prediction_type,debug=False):
    physicalFlag = False
    for subentity in entity.split(" "):
        try:
            result = ESCCheck(nlp,sen,subentity,dataset_path,tokenizer,args,wsd_model,prediction_type,debug)
            if result > 0:
                physicalFlag = True
        except:
            physicalFlag = True
            continue
            
    return physicalFlag


def GetTreeelement(input_text,file_name,debug=False):
    input_text_phrase=input_text.replace(" ",'+')
    input_text_phrase=input_text_phrase.replace(',','%2C')
#     url = 'http://trips.ihmc.us/parser/cgi/step?input=Tom+took+out+a+package+of+soup+from+the+pantry%2+making+sure+he+could+put+it+in+the+microwave&split-mode=split-sentences'
#     url = 'http://trips.ihmc.us/parser/cgi/drum?input=Tom+took+out+a+package+of+soup+from+the+pantry%2+making+sure+he+could+put+it+in+the+microwave&split-mode=split-sentences'
    url = 'http://trips.ihmc.us/parser/cgi/step-dev?input='+input_text_phrase+'&split-mode=split-sentences&number-parses-desired=1'
#     print(url)
    x = requests.get(url)
    savedStdout = sys.stdout 
    with open(file_name, 'w') as file:
        sys.stdout = file  #标准输出重定向至文件
        print(x.text)
    sys.stdout = savedStdout 
    xmlfilepath = os.path.abspath(file_name)
    tree = ET.parse(xmlfilepath)
    if debug:
        return tree,x
    else:
        return tree
    
def GetTreeelementFromFile(file_dic,example_id,single_story_id,sen_id):
    file_name=str(example_id)+"-"+str(single_story_id)+"-"+str(sen_id)+".xml"
    file_path = os.path.join(file_dic,example_id)
    if not os.path.exists(file_path):
        print("file doesn't exist")
    xmlfilepath = os.path.abspath(os.path.join(file_path,file_name))
    tree = ET.parse(xmlfilepath)
    return tree

def GetTreeelementFromFile2(file_name):
    if not os.path.exists(file_name):
        print("file doesn't exist")
    xmlfilepath = os.path.abspath(file_name)
    tree = ET.parse(xmlfilepath)
    return tree


def get_entity_list(sentences,segmenter):
    entity_list=[]
    filter_list=[]
    for sen in sentences:
        doc=segmenter(sen)
        pos_list=[t.pos_ for t in doc]
        token_list=[t.lemma_ for t in doc]
        count_tag=-1
        for i in range(len(pos_list)):
            if i>count_tag:
                if pos_list[i]=="NOUN":
                    temp_entity=""
                    for j in range(i,len(pos_list)):
                        if pos_list[j]=="NOUN":
                            temp_entity+=str(token_list[j])+" "
                            count_tag=j
                        else:
                            break
                    entity_list.append(temp_entity[:-1])
    entity_list=sorted(list(set(entity_list)))
    return entity_list


def GetEntityList(nlp,input_text,file_name,debug=False,input_file=False):
    ### remvoe 'IMPRO','IMRPO-SET', which are implicit
    NP_indicator=['THE','THE-SET','A','INDEF-SET','SM','PRO','PRO-SET','BARE','QUANTIFIER','WH-TERM','WH-TERM-SET']
    ### remove 'NEUTRAL' 'EXPERIENCER' , which are acasual
    ### remove 'FORMAL' , which are abstractions
    NP_role = ['AGENT','AFFECTED','FIGURE','GROUND','AFFECTED-RESULT','BENEFICIARY','NEUTRAL','EXPERIENCER','FORMAL'] 
    Modification_role = ['MOD','ASSOC-WITH']
    if input_file:
        tree = GetTreeelementFromFile2(file_name)
    else:
        tree=GetTreeelement(input_text,file_name=file_name)
    root = tree.getroot()
    uttlist=[]
    logic_form_list=[]
    VID_WORD_DICT={}
    Person_name=[]
    for utt in root.iter("utt"):
        uttlist.append(utt)
    # find all utt
    for utt in uttlist:
        terms = utt.find('terms')[1]
        for child in terms:
            child_dict={}
            child_dict['role']={}
            child_dict['ontology_id']=list(child.attrib.values())[0]
            for item in child:
                # {http://www.cs.rochester.edu/research/trips/LF#}word : {} : PUT-ON
                # {{http://www.cs.rochester.edu/research/trips/role#}AGENT : 
                # {'{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource': '#V89566'} : None
                if 'LF' in item.tag:
                    lf_term=item.tag.split('}')[-1]
                    child_dict[lf_term] = item.text
                else: # role
                    role_term=item.tag.split('}')[-1]
                    if len(item.attrib)!=0:
                        child_dict['role'][role_term] = list(item.attrib.values())[0].split('#')[-1]
                    else:
                        child_dict['role'][role_term] = item.text         
                if 'LEX' in child_dict['role'].keys():  ### remove () (North Pole)
                    child_dict['role']['LEX'] = child_dict['role']['LEX'].replace("(","").replace(")","").replace(" PUNC-MINUS ","-").upper()
                    
#---------------------------------------------- store single noun name-------------------------------------------------#
            if 'word' in child_dict.keys():
                if 'NAME-OF' in child_dict['role'].keys() or 'PERSON' in child_dict['type'] or 'CONSUMER' in child_dict['type']:  # Name of one person
#                     print(child_dict['word'])
                    if 'LEX' in child_dict['role'].keys(): ### Some word like his don't have LEX
                        VID_WORD_DICT[child_dict['ontology_id']]={'single_noun_name':input_text[input_text.upper().index(child_dict['role']['LEX']):\
                                                                                            input_text.upper().index(child_dict['role']['LEX'])+len(child_dict['role']['LEX'])]}
                    else:
                        VID_WORD_DICT[child_dict['ontology_id']]={'single_noun_name':input_text[input_text.upper().index(child_dict['word']):\
                                                                                            input_text.upper().index(child_dict['word'])+len(child_dict['word'])]}
                    #Person_name needs addition of lex
                    VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name_lex'] = VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name']
                    VID_WORD_DICT[child_dict['ontology_id']]['person'] = "person"
                    #Person_name can't be found in the physical check
                    Person_name.append(VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name'])
                else:
                    if child_dict['word'][-1]=="-":  ##[CO-,WOKERS]
                        VID_WORD_DICT[child_dict['ontology_id']]={'single_noun_name':child_dict['word'].replace("-","").lower()}
                    else:
                        VID_WORD_DICT[child_dict['ontology_id']]={'single_noun_name':child_dict['word'].replace("-"," ").lower()}
                    if 'LEX' in child_dict['role'].keys():
                        VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name_lex']=child_dict['role']['LEX'].replace("(","").replace(")","").lower()
#                         print(VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name_lex'])
                    else:
                        VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name_lex']=VID_WORD_DICT[child_dict['ontology_id']]['single_noun_name']
                VID_WORD_DICT[child_dict['ontology_id']]['start'] = int(child_dict['start'])
                VID_WORD_DICT[child_dict['ontology_id']]['end'] = int(child_dict['end'])
                if child_dict['indicator'] in ['PRO','PRO-SET','IMPRO','IMPRO-SET']:
                    VID_WORD_DICT[child_dict['ontology_id']]['pronoun'] = child_dict['indicator']
            else:# add content words without lex
                if 'LEX' in child_dict['role'].keys():
                    VID_WORD_DICT[child_dict['ontology_id']]={'single_noun_name_lex':child_dict['role']['LEX'].replace("(","").replace(")","").lower()}
                    VID_WORD_DICT[child_dict['ontology_id']]['start'] = int(child_dict['start'])
                    VID_WORD_DICT[child_dict['ontology_id']]['end'] = int(child_dict['end'])
            logic_form_list.append(child_dict)

            
    if debug:
        for item in VID_WORD_DICT:
            print(item)
            print(VID_WORD_DICT[item])
        for item in logic_form_list:
            print(item)


#---------------------------------------------- update and combine noun phrase --------------------------------------------#
    for item in logic_form_list:
        for key in Modification_role:
            if key in item['role'].keys():
                assoc_noun_id = item['role'][key]
                ### check whether it is in VID_DICT
                if assoc_noun_id in VID_WORD_DICT.keys() and item['ontology_id'] in VID_WORD_DICT.keys():
                    assoc_noun_name = VID_WORD_DICT[assoc_noun_id]['single_noun_name_lex']
                    item_noun_name = VID_WORD_DICT[item['ontology_id']]['single_noun_name_lex']
                    if VID_WORD_DICT[item['ontology_id']]['start'] == VID_WORD_DICT[assoc_noun_id]['start'] and \
                       VID_WORD_DICT[item['ontology_id']]['end'] == VID_WORD_DICT[assoc_noun_id]['end']:
                        if input_text.index(assoc_noun_name) < input_text.index(item_noun_name):
                            if (assoc_noun_name+" "+item_noun_name) in input_text:
                                VID_WORD_DICT[item['ontology_id']]['single_noun_name_lex'] = assoc_noun_name+" "+item_noun_name
                            if (assoc_noun_name+item_noun_name) in input_text:
                                VID_WORD_DICT[item['ontology_id']]['single_noun_name_lex'] = assoc_noun_name+item_noun_name
                        else:
                            if (item_noun_name+" "+assoc_noun_name) in input_text:
                                VID_WORD_DICT[item['ontology_id']]['single_noun_name_lex'] = item_noun_name+" "+assoc_noun_name
                            if (item_noun_name+assoc_noun_name) in input_text:
                                VID_WORD_DICT[item['ontology_id']]['single_noun_name_lex'] = item_noun_name+assoc_noun_name
                            


            
#----------------------------------------------- pick candidate C1 based on noun phrase-------------------------------------#
    candidate_entityId_list_C1=[]
    for item in logic_form_list:
        if item['indicator'] in NP_indicator or item['ontology_id'] in VID_WORD_DICT.keys() and 'person' in VID_WORD_DICT[item['ontology_id']].keys():
            candidate_entityId_list_C1.append(item['ontology_id'])
        # Consider SET 
        if item['indicator'] in ['THE-SET','INDEF-SET']:
            for key in item['role'].keys():
                if 'SEQUENCE' in key and item['role'][key] in VID_WORD_DICT.keys():
                    candidate_entityId_list_C1.append(item['role'][key])

    candidate_entityId_list_C1=list(set(candidate_entityId_list_C1))

#------------------------------------------------ pick candidate C2 based on change,comparation------------------------------#
    candidate_entityId_list_C2 = []
    for item in logic_form_list:
        for key in item['role'].keys(): ## some time key will be AGENT1
            for role_key in NP_role:
                if role_key in key:
                    candidate_entityId_list_C2.append(item['role'][key])
#     ### Consider SET
    for item in logic_form_list:
        if item['ontology_id'] in candidate_entityId_list_C2:
            for key in item['role'].keys():  
                if 'SEQUENCE' in key:
                    candidate_entityId_list_C2.append(item['role'][key])
    candidate_entityId_list_C2 = list(set(candidate_entityId_list_C1)&set(candidate_entityId_list_C2))


#------------------------------------------------ pick candidate C3 based on physical---------------------------------------------#
    candidate_entityId_list_C3 = []
    Physical_Dict={}
    for entity_id in candidate_entityId_list_C2:
        if entity_id in VID_WORD_DICT.keys() and 'pronoun' not in VID_WORD_DICT[entity_id].keys() and 'single_noun_name' in VID_WORD_DICT[entity_id].keys():
            if VID_WORD_DICT[entity_id]['single_noun_name'] not in Person_name:
                ### check physical from input dict or use model
                if CheckAbstract(nlp,input_text,VID_WORD_DICT[entity_id]['single_noun_name_lex'],\
                     dataset_path,tokenizer,args,wsd_model,prediction_type,debug=False):
                    candidate_entityId_list_C3.append(entity_id)
                    Physical_Dict[VID_WORD_DICT[entity_id]['single_noun_name_lex']]='physical'
                else:
                    Physical_Dict[VID_WORD_DICT[entity_id]['single_noun_name_lex']]='abstract'
            # add person 
            if VID_WORD_DICT[entity_id]['single_noun_name'] in Person_name:
                candidate_entityId_list_C3.append(entity_id)
                Physical_Dict[VID_WORD_DICT[entity_id]['single_noun_name']]='people'

    candidate_entityId_list_C3=list(set(candidate_entityId_list_C3))

    ### remove pronoun 
    final_entity_list= [VID_WORD_DICT[item]['single_noun_name_lex'] for item in candidate_entityId_list_C3\
                       if item in VID_WORD_DICT.keys() and 'pronoun' not in VID_WORD_DICT[item].keys()] 
    entity_id_dict={}
    for item in candidate_entityId_list_C3:
        if 'pronoun' not in VID_WORD_DICT[item].keys():
            entity_id_dict[VID_WORD_DICT[item]['single_noun_name_lex']]=item

    if debug:
        
        
        
        (candidate_entityId_list_C1)
        print(candidate_entityId_list_C2)
        print(candidate_entityId_list_C3)
        return final_entity_list,VID_WORD_DICT,logic_form_list,entity_id_dict,Physical_Dict
    else:
        return final_entity_list
    
    
def get_entity_list(sentences,segmenter):
    entity_list=[]
    filter_list=[]
    for sen in sentences:
        doc=segmenter(sen)
        pos_list=[t.pos_ for t in doc]
        token_list=[t.lemma_ for t in doc]
        count_tag=-1
        for i in range(len(pos_list)):
            if i>count_tag:
                if pos_list[i]=="NOUN":
                    temp_entity=""
                    for j in range(i,len(pos_list)):
                        if pos_list[j]=="NOUN":
                            temp_entity+=str(token_list[j])+" "
                            count_tag=j
                        else:
                            break
                    entity_list.append(temp_entity[:-1])
        count_tag=-1
        for i in range(len(pos_list)):
            if i>count_tag:
                if pos_list[i]=="PROPN":
                    temp_entity=""
                    for j in range(i,len(pos_list)):
                        if pos_list[j]=="PROPN":
                            temp_entity+=str(token_list[j])+" "
                            count_tag=j
                        else:
                            break
                    entity_list.append(temp_entity[:-1])
                    
    entity_list=sorted(list(set(entity_list)))
    return entity_list


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:                  
        os.makedirs(path)           
        print('New folder created')
    else:
        print("Folder exist")
        
        
#######################  CODAH ########################


def get_codah_dataset(file_name):
    codah_dataset=[]
    with open(file_name) as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for line in tsvreader:
            temp_dict={}
            temp_dict['type']=line[0]
            temp_dict['question']=line[1]
            temp_dict['answers']=line[2:6]
            temp_dict['label']=int(line[6])
            codah_dataset.append(temp_dict)
            
    for datasample in codah_dataset:
        stories=[]
        for item in datasample['answers']:
            stories.append(datasample['question']+" "+item)
        datasample['stories']=stories
        
        
    codah_dataset = segmentation_spacy_codah(codah_dataset,nlp,nlp_sen)
    
    return codah_dataset



def segmentation_spacy_codah(dataset,segmenter,nlp_sen):

    for sample_data in tqdm(dataset):
        sample_data['sentences']=[]
        sample_data['entity']=[]
        for story in sample_data['stories']:
            doc=nlp_sen(story)
            temp_story=[]
            for sen in doc.sents:
                text=sen.text.lstrip().rstrip()
                if len(text)>=3:
                    if text[-1]!=".":
                        if text[-1]=="?":
                            text=text[:-1]+"."
                        else:
                            text=text+"."
                temp_story.append(text.capitalize())

            sample_data['sentences'].append(temp_story)
    
    return dataset


def codah_post_preprocess(dev_dataset):

    for index,item in tqdm(enumerate(dev_dataset)):
        if len(item['entity'])==0:
            temp_entity_list=[]
            for temp_story in item['sentences']:
                temp_entity_list.append(get_entity_list(temp_story,nlp))
            item['entity']=temp_entity_list
        else:
            for en_index,entity_item in enumerate(item['entity']):
                if len(entity_item)==0:
                    item['entity'][en_index]=get_entity_list(item['sentences'][en_index],nlp)
                    
    for index,item in tqdm(enumerate(dev_dataset)):
        if sum([len(_) for _ in item['entity']])==0:

            temp_entity_list=[]
            for temp_story in item['sentences']:
                temp_entity_list.append(get_entity_list(temp_story,nlp))
            item['entity']=temp_entity_list
            
    for index,item in tqdm(enumerate(dev_dataset)):
#         try:
        entity_move_list_1=[]
        entity_move_list_2=[]
        entity_move_list_3=[]
        entity_move_list_4=[]
        for word in noise_word:
            for entity in item['entity'][0]:
                if entity.upper() == word.upper() :
                    entity_move_list_1.append(entity)
            for entity in item['entity'][1]:
                if entity.upper() == word.upper() :
                    entity_move_list_2.append(entity)
            for entity in item['entity'][2]:
                if entity.upper() == word.upper() :
                    entity_move_list_3.append(entity)
            for entity in item['entity'][3]:
                if entity.upper() == word.upper() :
                    entity_move_list_4.append(entity)

        item['entity'][0] = list(set(item['entity'][0]) - set(entity_move_list_1))
        item['entity'][1] = list(set(item['entity'][1]) - set(entity_move_list_2))
        item['entity'][2] = list(set(item['entity'][2]) - set(entity_move_list_3))
        item['entity'][3] = list(set(item['entity'][3]) - set(entity_move_list_4))
#         except:
#             print('error')
            
    return dev_dataset


####################### two-option  ########################

def remove_noise(dev_dataset):

    for index,item in tqdm(enumerate(dev_dataset)):
        try:
            entity_move_list_1=[]
            entity_move_list_2=[]
            for word in noise_word:
                for entity in item['entity_1']:
                    if entity.upper() == word.upper() :
                        entity_move_list_1.append(entity)
                for entity in item['entity_2']:
                    if entity.upper() == word.upper() :
                        entity_move_list_2.append(entity)


            item['entity_1'] = set(item['entity_1'] - set(entity_move_list_1))
            item['entity_2'] = set(item['entity_2'] - set(entity_move_list_2))
        except:continue
            
    return dev_dataset



def post_preprocess(dev_dataset):
    for index,item in tqdm(enumerate(dev_dataset)):
        if 'entity_1' not in item.keys():
            item['entity_1']=set(get_entity_list(item['goal_sol_1'],nlp))
        if 'entity_2' not in item.keys():
            item['entity_2']=set(get_entity_list(item['goal_sol_2'],nlp))
            
    for index,item in tqdm(enumerate(dev_dataset)):
        if len(item['entity_1'])==0:
            item['entity_1']=set(get_entity_list(item['goal_sol_1'],nlp))
        if len(item['entity_2'])==0:
            item['entity_2']=set(get_entity_list(item['goal_sol_2'],nlp))
            
    dev_dataset = remove_noise(dev_dataset)
    
    return dev_dataset
    
####################### ROCStories ########################

def get_roc_dataset(file_name):
    csv_reader = csv.reader(open(file_name))
    dev_dataset = []
    for index, line in tqdm(enumerate(csv_reader)):
        if index > 0:
            temp_dict = {
                'sentence': line[1:5],
                'end1': line[5],
                'end2': line[6],
                'goal_sol_1':line[1:6],
                'goal_sol_2':line[1:5]+[line[6]],
                'label': int(line[7])
            }
            dev_dataset.append(temp_dict)
            
    return dev_dataset


####################### PIQA ########################

def segmentation_spacy_piqa(dataset,segmenter,nlp_sen):

    for sample_data in tqdm(dataset):
        # sentence1
        sentence1=sample_data['sol1']
        doc=nlp_sen(sentence1)
        goal=sample_data['goal'].lstrip().rstrip()
        if len(goal)>=3:
            if goal[-1] != ".":
                if goal[-1]=="?":
                    goal=goal[:-1]+"."
                else:
                    goal=goal+"."
        goal_sol_1=[goal.capitalize()]
        for sen in doc.sents:
            text=sen.text.lstrip().rstrip()
            if len(text)>=3:
                if text[-1]!=".":
                    if text[-1]=="?":
                        text=text[:-1]+"."
                    else:
                        text=text+"."
                goal_sol_1.append(text.capitalize())

        sample_data['goal_sol_1']=goal_sol_1
#         sample_data['entity_1']=get_entity_list(goal_sol_1,segmenter)

        # sentence2
        sentence2=sample_data['sol2']
        doc=nlp_sen(sentence2)
        goal_sol_2=[goal.capitalize()]
        for sen in doc.sents:
            text=sen.text.lstrip().rstrip()
            if len(text)>=3:
                if text[-1]!=".":
                    if text[-1]=="?":
                        text=text[:-1]+"."
                    else:
                        text=text+"."
                goal_sol_2.append(text.capitalize())

        sample_data['goal_sol_2']=goal_sol_2
#         sample_data['entity_2']=get_entity_list(goal_sol_2,segmenter) 
    
    return dataset


def get_piqa_dataset(input_name,label_name,segmenter,nlp_sen):
    with open(input_name, 'r') as json_file:
        dev_data_list = list(json_file)
    
    with open(label_name, 'r') as json_file:
        dev_label_list = list(json_file)

    dev_dataset=[]
    for dev_data,dev_label in zip(dev_data_list,dev_label_list):
        data = json.loads(dev_data)
        label = json.loads(dev_label)
        data['label']=label
        dev_dataset.append(data)
        
    dev_dataset = segmentation_spacy_piqa(dev_dataset,segmenter,nlp_sen)
        
    return dev_dataset


####################### ANLI ########################

def get_anli_dataset(input_name,label_name,segmenter,nlp_sen):
    with open(input_name, 'r') as json_file:
        dev_data_list = list(json_file)
    
    with open(label_name, 'r') as json_file:
        dev_label_list = list(json_file)

    dev_dataset=[]
    for dev_data,dev_label in zip(dev_data_list,dev_label_list):
        data = json.loads(dev_data)
        label = json.loads(dev_label)
        data['label']=label
        dev_dataset.append(data)
        
    dev_dataset = segmentation_spacy_anli(dev_dataset,segmenter,nlp_sen)
        
    return dev_dataset


def segmentation_spacy_anli(dataset,segmenter,nlp_sen):

    for sample_data in tqdm(dataset):

        obs1 = nlp_sen(sample_data['obs1'])
        obs2 = nlp_sen(sample_data['obs2'])
        hyp1 = nlp_sen(sample_data['hyp1'])
        hyp2 = nlp_sen(sample_data['hyp2'])
        
        goal_sol_1=[]
        goal_sol_2=[]
        for sen in obs1.sents:
            text=sen.text.lstrip().rstrip()
            if len(text)>=3:
                if text[-1]!=".":
                    text=text+"."
                goal_sol_1.append(text)
                goal_sol_2.append(text)
                
        for sen in hyp1.sents:
            text=sen.text.lstrip().rstrip()
            if len(text)>=3:
                if text[-1]!=".":
                    text=text+"."
                goal_sol_1.append(text)        
                
        for sen in hyp2.sents:
            text=sen.text.lstrip().rstrip()
            if len(text)>=3:
                if text[-1]!=".":
                    text=text+"."
                goal_sol_2.append(text)  

        for sen in obs2.sents:
            text=sen.text.lstrip().rstrip()
            if len(text)>=3:
                if text[-1]!=".":
                    text=text+"."
                goal_sol_1.append(text)
                goal_sol_2.append(text)
                
                
        sample_data['goal_sol_1']=goal_sol_1
        sample_data['goal_sol_2']=goal_sol_2
    
    return dataset