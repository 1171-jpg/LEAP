import time
import torch
from www.utils import format_time
import numpy as np
from transformers import RobertaForMultipleChoice
import progressbar
from www.model.evalAdaptEntity import evaluate_tiered
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss


# Train a PyTorch model for one epoch
def train_epoch(model, optimizer, train_dataloader, device, list_output=False, num_outputs=1, span_mode=False, seg_mode=False, classifier=None, multitask_idx=None):
  t0 = time.time()

  if not list_output:
    total_loss = 0
  else:
    total_loss = [0 for _ in range(num_outputs)]

  # Training mode
  model.train()

  if len(train_dataloader) * train_dataloader.batch_size >= 2500:
    progress_update = True
  else:
    progress_update = False

  for step, batch in enumerate(train_dataloader):
    # Progress update
    if progress_update and step % 50 == 0 and not step == 0:
      elapsed = format_time(time.time() - t0)
      print('\t(%s) Starting batch %s of %s.' % (elapsed, str(step), str(len(train_dataloader))))

    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    labels = batch[2].to(device)

    # if input_ids.dim() > 2:
    #   input_ids = input_ids.view(input_ids.shape[0], -1)
    #   input_mask = input_mask.view(input_mask.shape[0], -1)

    # In some cases, we also include a span for each training sequence which the model uses to classify only certain parts of the input
    if span_mode:
      spans = batch[3].to(device)
    elif seg_mode:
      segment_ids = batch[3].to(device)
    else:
      spans = None

    # Forward pass
    model.zero_grad()
    if multitask_idx == None:
      if span_mode:
        out = model(input_ids, 
              token_type_ids=None, 
              attention_mask=input_mask, 
              labels=labels,
              spans=spans)
      elif seg_mode:
        out = model(input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    labels=labels)
      else:      
        out = model(input_ids, 
                    token_type_ids=None, 
                    attention_mask=input_mask,
                    labels=labels)
    else:
      if span_mode:
        out = model(input_ids, 
              token_type_ids=None, 
              attention_mask=input_mask, 
              labels=labels,
              spans=spans,
              task_idx=multitask_idx)
      elif seg_mode:
        out = model(input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    labels=labels,
                    task_idx=multitask_idx)
      else:      
        out = model(input_ids, 
                    token_type_ids=None, 
                    attention_mask=input_mask,
                    labels=labels,
                    task_idx=multitask_idx)                    

    if classifier != None:
      sequence_output = out[0]
      logits = classifier(out)

      loss = None
      if labels is not None:
          if self.num_labels == 1:
              #  We are doing regression
              loss_fct = MSELoss()
              loss = loss_fct(logits.view(-1), labels.view(-1))
          elif self.num_labels == 2:
              loss_fct = CrossEntropyLoss()
              loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
              
    else:
      loss = out[0]

    # Backward pass
    if not list_output:
      total_loss += loss.item()
      loss.backward()
    else:
      for o in range(num_outputs):
        total_loss[o] += loss[o].item()
        loss[o].backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping

    optimizer.step()
  
  if list_output:
    return list(np.array(total_loss) / len(train_dataloader)), model
  else:
    return total_loss / len(train_dataloader), model



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


# Train a state classification pipeline for one epoch
def train_epoch_tiered(maxStoryLength,max_story_length,num_attributes,trip_model,trip_optimizer,train_dataloader, device, seg_mode=False, return_losses=False, build_learning_curves=False, val_dataloader=None, train_lc_data=None, val_lc_data=None):
  t0 = time.time()

  total_loss = 0

  # Training mode

  trip_model.train() 
#   trip_model.tslm.train() 
  for layer in trip_model.precondition_classifiers:
    layer.train()
  for layer in trip_model.effect_classifiers:
    layer.train()    

  # if len(train_dataloader) * train_dataloader.batch_size >= 2500:
  #   progress_update = True
  # else:
  #   progress_update = False
  progress_update = False

  bar_size = len(train_dataloader)
  bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
  bar_idx = 0
  bar.start()

  if train_lc_data is not None:
    train_lc_data.append([])
  if val_lc_data is not None:
    val_lc_data.append([])

  for step, batch in enumerate(train_dataloader):
    # Progress update
    if progress_update and step % 50 == 0 and not step == 0:
      elapsed = format_time(time.time() - t0)
      print('\t(%s) Starting batch %s of %s.' % (elapsed, str(step), str(len(train_dataloader))))

    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device) #.to(torch.int64).to('cpu')
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    timestep_type_ids=batch[9].long().to(device)
    if seg_mode:
      segment_ids = batch[8].to(device)
    else:
      segment_ids = None
    
    modify_input_ids = batch[0].view(-1, max_story_length, maxStoryLength).long().to(device)
    modify_input_mask = batch[3].view(-1, max_story_length, maxStoryLength).to(device)
    modify_preconditions = batch[5].view(-1, max_story_length,
                                                num_attributes).long().to(device)
    modify_effects = batch[6].view(-1, max_story_length,
                                                num_attributes).long().to(device)
    modify_conflicts = batch[7].view(-1, max_story_length).long().to(device)
    modify_timestep_type_ids=batch[9].view(-1, max_story_length, maxStoryLength).long().to(device)


    # Forward pass
    
    # Newly modified
    prep_result=torch.tensor([]).to(device)
    effe_result=torch.tensor([]).to(device)
    conflict_result=torch.tensor([]).to(device)
    total_entity_loss_preconditions=0
    total_entity_loss_effect=0
    total_entity_loss_conflicts=0
    total_entity_loss_stories=0
    total_entity_loss_totals=0
    
    for entity_idx in range(len(modify_input_ids)):
        trip_model.zero_grad()
        entity_input = modify_input_ids[entity_idx]
        entity_mask = modify_input_mask[entity_idx]
        entity_timestep = modify_timestep_type_ids[entity_idx]
        entity_preconditions = modify_preconditions[entity_idx]
        entity_effects = modify_effects[entity_idx]
        entity_confict=modify_conflicts[entity_idx]
        entity_length=input_lengths.view(-1)[entity_idx]
        sentense_mask=torch.tensor((entity_length*[1]+(max_story_length-entity_length)*[0])).to(device) 
        # loss    
        tempout=trip_model(entity_input,
          input_lengths,
          input_entities,
          entity_timestep,
          sentense_mask,         
          attention_mask=entity_mask,
          token_type_ids=segment_ids,
          attributes=attributes,
          preconditions=entity_preconditions,
          effects=entity_effects,
          conflicts=entity_confict,
          labels=labels,
          training=True)
        
        loss = tempout['total_loss']
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(trip_model.parameters(), 1.0) # Gradient clipping

        trip_optimizer.step()
        
        #store and update value
        prep_result=torch.cat((prep_result,tempout['out_preconditions']),dim=0)
        effe_result=torch.cat((effe_result,tempout['out_effects']),dim=0)
        conflict_result=torch.cat((conflict_result,tempout['out_conflicts']),dim=0)
        total_entity_loss_preconditions+=tempout['loss_preconditions']
        total_entity_loss_effect+=tempout['loss_effects']
        total_entity_loss_conflicts+=tempout['loss_conflicts']
        total_entity_loss_totals+=tempout['total_loss']
    
    # summarize data
    batch_size, num_stories, num_entities, num_sents, seq_length = input_ids.shape
    prep_result,effe_result=update_result(prep_result, effe_result, input_ids, input_lengths,
                  input_entities,num_attributes)
    out={}
    out['out_preconditions']=prep_result
    out['loss_preconditions']=total_entity_loss_preconditions/len(modify_input_ids)
    out['out_effects']=effe_result
    out['loss_effects']=total_entity_loss_effect/len(modify_input_ids)
    out['out_conflicts']=conflict_result
    out['loss_conflicts']=total_entity_loss_conflicts/len(modify_input_ids)
    out['total_loss']=total_entity_loss_totals/len(modify_input_ids)
    # add story information
    story_out=out['out_conflicts'].view(batch_size, num_stories, num_entities, -1) 
    story_out = -torch.sum(story_out, dim=(2,3)) / 2
    out['out_stories']=story_out          
    loss_fct = CrossEntropyLoss()
    loss_stories = loss_fct(story_out, labels)
    out['loss_stories']=loss_stories 
    
    # Build learning curve data if needed
    if build_learning_curves:
      train_record = {'epoch': len(train_lc_data) - 1,
                      'iteration': (len(train_lc_data) - 1) * len(train_dataloader) + step,
                      'loss_preconditions': float(out['loss_preconditions'].detach().cpu().numpy()) / trip_model.num_attributes,
                      'loss_effects': float(out['loss_effects'].detach().cpu().numpy()) / trip_model.num_attributes,
                      'loss_conflicts': float(out['loss_conflicts'].detach().cpu().numpy()),
                      'loss_stories': float(out['loss_stories'].detach().cpu().numpy()),
                      'loss_total': float(out['total_loss'].detach().cpu().numpy())}
      train_lc_data[-1].append(train_record)

      # Add a validation record 5 times per epoch
#       chunk_size = len(train_dataloader) // 3
#       if (len(train_dataloader) - step - 1) % chunk_size == 0:
#         validation_results = evaluate_tiered(maxStoryLength,max_story_length,num_attributes,trip_model, val_dataloader, device, [(accuracy_score, 'accuracy'), (f1_score, 'f1')], seg_mode=False, return_explanations=True, return_losses=True, verbose=True)
#         out = validation_results[16]

#         val_record = {'epoch': len(val_lc_data) - 1,
#                       'iteration': (len(val_lc_data) - 1) * len(train_dataloader) + step,
#                       'loss_preconditions': float(out['loss_preconditions'].detach().cpu().numpy()) / trip_model.num_attributes,
#                       'loss_effects': float(out['loss_effects'].detach().cpu().numpy()) / trip_model.num_attributes,
#                       'loss_conflicts': float(out['loss_conflicts'].detach().cpu().numpy()),
#                       'loss_stories': float(out['loss_stories'].detach().cpu().numpy()),
#                       'loss_total': float(out['total_loss'].detach().cpu().numpy())}
#         val_lc_data[-1].append(val_record)

    
    bar_idx += 1
    bar.update(bar_idx)

  bar.finish()
  
  return total_loss / len(train_dataloader)

