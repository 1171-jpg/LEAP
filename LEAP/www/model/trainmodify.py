import time
import torch
from www.utils import format_time
import numpy as np
from transformers import RobertaForMultipleChoice
import progressbar
from www.model.evalmodify import evaluate_tiered
from sklearn.metrics import accuracy_score, f1_score



# Train a PyTorch model for one epoch
def ComputeLoss(out1,out2):
    loss_weights=[0.0, 0.4, 0.4, 0.2, 0.0]
    total_loss = 0.0
    total_loss += loss_weights[1] * out1['loss_preconditions'] / 20
    total_loss += loss_weights[2] * out1['loss_effects']  / 20
    total_loss += loss_weights[3] * out2['loss_conflicts']
    total_loss += loss_weights[4] * out2['loss_stories']

    return total_loss
def train_epoch(model,
                optimizer,
                train_dataloader,
                device,
                list_output=False,
                num_outputs=1,
                span_mode=False,
                seg_mode=False,
                classifier=None,
                multitask_idx=None):
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
            print('\t(%s) Starting batch %s of %s.' %
                  (elapsed, str(step), str(len(train_dataloader))))

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
                    loss = loss_fct(logits.view(-1, self.num_labels),
                                    labels.view(-1))

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

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       1.0)  # Gradient clipping

        optimizer.step()

    if list_output:
        return list(np.array(total_loss) / len(train_dataloader)), model
    else:
        return total_loss / len(train_dataloader), model


# Train a state classification pipeline for one epoch
def train_epoch_tiered(model1,
                       model2,
                       optimizer1,
                       optimizer2,
                       train_dataloader,
                       device,
                       seg_mode=False,
                       return_losses=False,
                       build_learning_curves=False,
                       val_dataloader=None,
                       train_lc_data=None,
                       val_lc_data=None):
    t0 = time.time()

    total_loss = 0

    # Training mode
    model1.train()
    model2.train()
    for layer in model1.precondition_classifiers:
        layer.train()
    for layer in model1.effect_classifiers:
        layer.train()

    # if len(train_dataloader) * train_dataloader.batch_size >= 2500:
    #   progress_update = True
    # else:
    #   progress_update = False
    progress_update = False

    bar_size = len(train_dataloader)
    bar = progressbar.ProgressBar(max_value=bar_size,
                                  widgets=[
                                      progressbar.Bar('#', '[', ']'), ' ',
                                      progressbar.Percentage()
                                  ])
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
            print('\t(%s) Starting batch %s of %s.' %
                  (elapsed, str(step), str(len(train_dataloader))))

        input_ids = batch[0].long().to(device)
        input_lengths = batch[1].to(device)  #.to(torch.int64).to('cpu')
        input_entities = batch[2].to(device)
        input_mask = batch[3].to(device)
        attributes = batch[4].long().to(device)
        preconditions = batch[5].long().to(device)
        effects = batch[6].long().to(device)
        conflicts = batch[7].long().to(device)
        labels = batch[8].long().to(device)

        if seg_mode:
            segment_ids = batch[8].to(device)
        else:
            segment_ids = None

        # Forward pass
        model1.zero_grad()
        model2.zero_grad()
        out_1 = model1(input_ids, 
                    input_lengths,
                    input_entities,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    attributes=attributes,
                    preconditions=preconditions,
                    effects=effects,
                    training=True)

        out_preconditions_softmax=out_1['out_preconditions_softmax']
        out_effects_softmax=out_1['out_effects_softmax']
        outcls=out_1['out']
        out_2 = model2(input_ids, 
                    input_lengths,
                    input_entities,
                    out=outcls,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    attributes=attributes,
                    out_preconditions_softmax=out_preconditions_softmax,
                    out_effects_softmax=out_effects_softmax,
                    conflicts=conflicts,
                    labels=labels,
                    training=True)

        out={}
        for k in out_1:
            out[k]=out_1[k]
        for k in out_2:
            out[k]=out_2[k]
        out['total_loss']=ComputeLoss(out_1,out_2)
        
        loss = out['total_loss']

        # Backward pass
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model1.parameters(),
                                       1.0)  # Gradient clipping

        optimizer1.step()
        optimizer2.step()

        # Build learning curve data if needed
        if build_learning_curves:
            train_record = {
                'epoch':
                len(train_lc_data) - 1,
                'iteration':
                (len(train_lc_data) - 1) * len(train_dataloader) + step,
                'loss_preconditions':
                float(out['loss_preconditions'].detach().cpu().numpy()) /
                model1.num_attributes,
                'loss_effects':
                float(out['loss_effects'].detach().cpu().numpy()) /
                model1.num_attributes,
                'loss_conflicts':
                float(out['loss_conflicts'].detach().cpu().numpy()),
                'loss_stories':
                float(out['loss_stories'].detach().cpu().numpy()),
                'loss_total':
                float(out['total_loss'].detach().cpu().numpy())
            }
            train_lc_data[-1].append(train_record)

            # Add a validation record 5 times per epoch
            chunk_size = len(train_dataloader) // 5
            if (len(train_dataloader) - step - 1) % chunk_size == 0:
                validation_results = evaluate_tiered(
                    model1,
                    model2,
                    val_dataloader,
                    device, [(accuracy_score, 'accuracy'), (f1_score, 'f1')],
                    seg_mode=False,
                    return_explanations=True,
                    return_losses=True,
                    verbose=False)
                out = validation_results[16]

                val_record = {
                    'epoch':
                    len(val_lc_data) - 1,
                    'iteration':
                    (len(val_lc_data) - 1) * len(train_dataloader) + step,
                    'loss_preconditions':
                    float(out['loss_preconditions'].detach().cpu().numpy()) /
                    model1.num_attributes,
                    'loss_effects':
                    float(out['loss_effects'].detach().cpu().numpy()) /
                    model1.num_attributes,
                    'loss_conflicts':
                    float(out['loss_conflicts'].detach().cpu().numpy()),
                    'loss_stories':
                    float(out['loss_stories'].detach().cpu().numpy()),
                    'loss_total':
                    float(out['total_loss'].detach().cpu().numpy())
                }
                val_lc_data[-1].append(val_record)

        bar_idx += 1
        bar.update(bar_idx)

    bar.finish()

    return total_loss / len(train_dataloader), model1