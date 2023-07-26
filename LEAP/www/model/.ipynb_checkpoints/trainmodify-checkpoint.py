{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ee185f3e",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'www'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-5b7c7251d7c2>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mwww\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mutils\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mformat_time\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mtransformers\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mRobertaForMultipleChoice\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'www'"
     ]
    }
   ],
   "source": [
    "import time\n",
    "import torch\n",
    "from www.utils import format_time\n",
    "import numpy as np\n",
    "from transformers import RobertaForMultipleChoice\n",
    "import progressbar\n",
    "from www.model.eval import evaluate_tiered\n",
    "from sklearn.metrics import accuracy_score, f1_score\n",
    "\n",
    "\n",
    "\n",
    "# Train a PyTorch model for one epoch\n",
    "def ComputeLoss(out1,out2):\n",
    "    loss_weights=[0.0, 0.4, 0.4, 0.2, 0.0]\n",
    "    total_loss = 0.0\n",
    "    total_loss += loss_weights[1] * out1['loss_preconditions'] / 20\n",
    "    total_loss += loss_weights[2] * out1['loss_effects']  / 20\n",
    "    total_loss += loss_weights[3] * out2['loss_conflicts']\n",
    "    total_loss += loss_weights[4] * out2['loss_stories']\n",
    "\n",
    "    return total_loss\n",
    "def train_epoch(model,\n",
    "                optimizer,\n",
    "                train_dataloader,\n",
    "                device,\n",
    "                list_output=False,\n",
    "                num_outputs=1,\n",
    "                span_mode=False,\n",
    "                seg_mode=False,\n",
    "                classifier=None,\n",
    "                multitask_idx=None):\n",
    "    t0 = time.time()\n",
    "\n",
    "    if not list_output:\n",
    "        total_loss = 0\n",
    "    else:\n",
    "        total_loss = [0 for _ in range(num_outputs)]\n",
    "\n",
    "    # Training mode\n",
    "    model.train()\n",
    "\n",
    "    if len(train_dataloader) * train_dataloader.batch_size >= 2500:\n",
    "        progress_update = True\n",
    "    else:\n",
    "        progress_update = False\n",
    "\n",
    "    for step, batch in enumerate(train_dataloader):\n",
    "        # Progress update\n",
    "        if progress_update and step % 50 == 0 and not step == 0:\n",
    "            elapsed = format_time(time.time() - t0)\n",
    "            print('\\t(%s) Starting batch %s of %s.' %\n",
    "                  (elapsed, str(step), str(len(train_dataloader))))\n",
    "\n",
    "        input_ids = batch[0].to(device)\n",
    "        input_mask = batch[1].to(device)\n",
    "        labels = batch[2].to(device)\n",
    "\n",
    "        # if input_ids.dim() > 2:\n",
    "        #   input_ids = input_ids.view(input_ids.shape[0], -1)\n",
    "        #   input_mask = input_mask.view(input_mask.shape[0], -1)\n",
    "\n",
    "        # In some cases, we also include a span for each training sequence which the model uses to classify only certain parts of the input\n",
    "        if span_mode:\n",
    "            spans = batch[3].to(device)\n",
    "        elif seg_mode:\n",
    "            segment_ids = batch[3].to(device)\n",
    "        else:\n",
    "            spans = None\n",
    "\n",
    "        # Forward pass\n",
    "        model.zero_grad()\n",
    "        if multitask_idx == None:\n",
    "            if span_mode:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=None,\n",
    "                            attention_mask=input_mask,\n",
    "                            labels=labels,\n",
    "                            spans=spans)\n",
    "            elif seg_mode:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=segment_ids,\n",
    "                            attention_mask=input_mask,\n",
    "                            labels=labels)\n",
    "            else:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=None,\n",
    "                            attention_mask=input_mask,\n",
    "                            labels=labels)\n",
    "        else:\n",
    "            if span_mode:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=None,\n",
    "                            attention_mask=input_mask,\n",
    "                            labels=labels,\n",
    "                            spans=spans,\n",
    "                            task_idx=multitask_idx)\n",
    "            elif seg_mode:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=segment_ids,\n",
    "                            attention_mask=input_mask,\n",
    "                            labels=labels,\n",
    "                            task_idx=multitask_idx)\n",
    "            else:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=None,\n",
    "                            attention_mask=input_mask,\n",
    "                            labels=labels,\n",
    "                            task_idx=multitask_idx)\n",
    "\n",
    "        if classifier != None:\n",
    "            sequence_output = out[0]\n",
    "            logits = classifier(out)\n",
    "\n",
    "            loss = None\n",
    "            if labels is not None:\n",
    "                if self.num_labels == 1:\n",
    "                    #  We are doing regression\n",
    "                    loss_fct = MSELoss()\n",
    "                    loss = loss_fct(logits.view(-1), labels.view(-1))\n",
    "                elif self.num_labels == 2:\n",
    "                    loss_fct = CrossEntropyLoss()\n",
    "                    loss = loss_fct(logits.view(-1, self.num_labels),\n",
    "                                    labels.view(-1))\n",
    "\n",
    "        else:\n",
    "            loss = out[0]\n",
    "\n",
    "        # Backward pass\n",
    "        if not list_output:\n",
    "            total_loss += loss.item()\n",
    "            loss.backward()\n",
    "        else:\n",
    "            for o in range(num_outputs):\n",
    "                total_loss[o] += loss[o].item()\n",
    "                loss[o].backward()\n",
    "\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(),\n",
    "                                       1.0)  # Gradient clipping\n",
    "\n",
    "        optimizer.step()\n",
    "\n",
    "    if list_output:\n",
    "        return list(np.array(total_loss) / len(train_dataloader)), model\n",
    "    else:\n",
    "        return total_loss / len(train_dataloader), model\n",
    "\n",
    "\n",
    "# Train a state classification pipeline for one epoch\n",
    "def train_epoch_tiered(model1,\n",
    "                       model2,\n",
    "                       optimizer,\n",
    "                       train_dataloader,\n",
    "                       device,\n",
    "                       seg_mode=False,\n",
    "                       return_losses=False,\n",
    "                       build_learning_curves=False,\n",
    "                       val_dataloader=None,\n",
    "                       train_lc_data=None,\n",
    "                       val_lc_data=None):\n",
    "    t0 = time.time()\n",
    "\n",
    "    total_loss = 0\n",
    "\n",
    "    # Training mode\n",
    "    model1.train()\n",
    "    model2.train()\n",
    "    for layer in model1.precondition_classifiers:\n",
    "        layer.train()\n",
    "    for layer in model1.effect_classifiers:\n",
    "        layer.train()\n",
    "\n",
    "    # if len(train_dataloader) * train_dataloader.batch_size >= 2500:\n",
    "    #   progress_update = True\n",
    "    # else:\n",
    "    #   progress_update = False\n",
    "    progress_update = False\n",
    "\n",
    "    bar_size = len(train_dataloader)\n",
    "    bar = progressbar.ProgressBar(max_value=bar_size,\n",
    "                                  widgets=[\n",
    "                                      progressbar.Bar('#', '[', ']'), ' ',\n",
    "                                      progressbar.Percentage()\n",
    "                                  ])\n",
    "    bar_idx = 0\n",
    "    bar.start()\n",
    "\n",
    "    if train_lc_data is not None:\n",
    "        train_lc_data.append([])\n",
    "    if val_lc_data is not None:\n",
    "        val_lc_data.append([])\n",
    "\n",
    "    for step, batch in enumerate(train_dataloader):\n",
    "        # Progress update\n",
    "        if progress_update and step % 50 == 0 and not step == 0:\n",
    "            elapsed = format_time(time.time() - t0)\n",
    "            print('\\t(%s) Starting batch %s of %s.' %\n",
    "                  (elapsed, str(step), str(len(train_dataloader))))\n",
    "\n",
    "        input_ids = batch[0].long().to(device)\n",
    "        input_lengths = batch[1].to(device)  #.to(torch.int64).to('cpu')\n",
    "        input_entities = batch[2].to(device)\n",
    "        input_mask = batch[3].to(device)\n",
    "        attributes = batch[4].long().to(device)\n",
    "        preconditions = batch[5].long().to(device)\n",
    "        effects = batch[6].long().to(device)\n",
    "        conflicts = batch[7].long().to(device)\n",
    "        labels = batch[8].long().to(device)\n",
    "\n",
    "        if seg_mode:\n",
    "            segment_ids = batch[8].to(device)\n",
    "        else:\n",
    "            segment_ids = None\n",
    "\n",
    "        # Forward pass\n",
    "        model1.zero_grad()\n",
    "        model2.zero_grad()\n",
    "        out_1 = model1(input_ids, \n",
    "                    input_lengths,\n",
    "                    input_entities,\n",
    "                    attention_mask=input_mask,\n",
    "                    token_type_ids=segment_ids,\n",
    "                    attributes=attributes,\n",
    "                    preconditions=preconditions,\n",
    "                    effects=effects,\n",
    "                    training=True)\n",
    "\n",
    "        out_preconditions_softmax=out_1['out_preconditions_softmax']\n",
    "        out_effects_softmax=out_1['out_effects_softmax']\n",
    "        outcls=out_1['out']\n",
    "        out_2 = model2(input_ids, \n",
    "                    input_lengths,\n",
    "                    input_entities,\n",
    "                    out=outcls,\n",
    "                    attention_mask=input_mask,\n",
    "                    token_type_ids=segment_ids,\n",
    "                    attributes=attributes,\n",
    "                    out_preconditions_softmax=out_preconditions_softmax,\n",
    "                    out_effects_softmax=out_effects_softmax,\n",
    "                    conflicts=conflicts,\n",
    "                    labels=labels,\n",
    "                    training=True)\n",
    "\n",
    "        out={}\n",
    "        for k in out_1:\n",
    "            out[k]=out_1[k]\n",
    "        for k in out_2:\n",
    "            out[k]=out_2[k]\n",
    "        out['total_loss']=ComputeLoss(out_1,out_2)\n",
    "        \n",
    "        loss = out['total_loss']\n",
    "\n",
    "        # Backward pass\n",
    "        total_loss += loss.item()\n",
    "        loss.backward()\n",
    "\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(),\n",
    "                                       1.0)  # Gradient clipping\n",
    "\n",
    "        optimizer.step()\n",
    "\n",
    "        # Build learning curve data if needed\n",
    "        if build_learning_curves:\n",
    "            train_record = {\n",
    "                'epoch':\n",
    "                len(train_lc_data) - 1,\n",
    "                'iteration':\n",
    "                (len(train_lc_data) - 1) * len(train_dataloader) + step,\n",
    "                'loss_preconditions':\n",
    "                float(out['loss_preconditions'].detach().cpu().numpy()) /\n",
    "                model.num_attributes,\n",
    "                'loss_effects':\n",
    "                float(out['loss_effects'].detach().cpu().numpy()) /\n",
    "                model.num_attributes,\n",
    "                'loss_conflicts':\n",
    "                float(out['loss_conflicts'].detach().cpu().numpy()),\n",
    "                'loss_stories':\n",
    "                float(out['loss_stories'].detach().cpu().numpy()),\n",
    "                'loss_total':\n",
    "                float(out['total_loss'].detach().cpu().numpy())\n",
    "            }\n",
    "            train_lc_data[-1].append(train_record)\n",
    "\n",
    "            # Add a validation record 5 times per epoch\n",
    "            chunk_size = len(train_dataloader) // 5\n",
    "            if (len(train_dataloader) - step - 1) % chunk_size == 0:\n",
    "                validation_results = evaluate_tiered(\n",
    "                    model,\n",
    "                    val_dataloader,\n",
    "                    device, [(accuracy_score, 'accuracy'), (f1_score, 'f1')],\n",
    "                    seg_mode=False,\n",
    "                    return_explanations=True,\n",
    "                    return_losses=True,\n",
    "                    verbose=False)\n",
    "                out = validation_results[16]\n",
    "\n",
    "                val_record = {\n",
    "                    'epoch':\n",
    "                    len(val_lc_data) - 1,\n",
    "                    'iteration':\n",
    "                    (len(val_lc_data) - 1) * len(train_dataloader) + step,\n",
    "                    'loss_preconditions':\n",
    "                    float(out['loss_preconditions'].detach().cpu().numpy()) /\n",
    "                    model.num_attributes,\n",
    "                    'loss_effects':\n",
    "                    float(out['loss_effects'].detach().cpu().numpy()) /\n",
    "                    model.num_attributes,\n",
    "                    'loss_conflicts':\n",
    "                    float(out['loss_conflicts'].detach().cpu().numpy()),\n",
    "                    'loss_stories':\n",
    "                    float(out['loss_stories'].detach().cpu().numpy()),\n",
    "                    'loss_total':\n",
    "                    float(out['total_loss'].detach().cpu().numpy())\n",
    "                }\n",
    "                val_lc_data[-1].append(val_record)\n",
    "\n",
    "        bar_idx += 1\n",
    "        bar.update(bar_idx)\n",
    "\n",
    "    bar.finish()\n",
    "\n",
    "    return total_loss / len(train_dataloader), model"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
