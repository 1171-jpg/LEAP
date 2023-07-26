{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92cb1d63",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import time\n",
    "from www.utils import format_time, read_tsv\n",
    "import torch\n",
    "import json\n",
    "import os\n",
    "import progressbar\n",
    "from www.dataset.ann import att_default_values\n",
    "\n",
    "\n",
    "# Run evaluation for a PyTorch model\n",
    "\n",
    "\n",
    "def ComputeLoss(out1,out2):\n",
    "    loss_weights=[0.0, 0.4, 0.4, 0.2, 0.0]\n",
    "    total_loss = 0.0\n",
    "    total_loss += loss_weights[1] * out1['loss_preconditions'] / 20\n",
    "    total_loss += loss_weights[2] * out1['loss_effects']  / 20\n",
    "    total_loss += loss_weights[3] * out2['loss_conflicts']\n",
    "    total_loss += loss_weights[4] * out2['loss_stories']\n",
    "\n",
    "    return total_loss\n",
    "\n",
    "def evaluate(model,\n",
    "             eval_dataloader,\n",
    "             device,\n",
    "             metrics,\n",
    "             list_output=False,\n",
    "             num_outputs=1,\n",
    "             span_mode=False,\n",
    "             seg_mode=False,\n",
    "             return_softmax=False,\n",
    "             multilabel=False,\n",
    "             lm=False):\n",
    "    print('\\tBeginning evaluation...')\n",
    "\n",
    "    t0 = time.time()\n",
    "\n",
    "    model.zero_grad()\n",
    "    model.eval()\n",
    "\n",
    "    all_labels = None\n",
    "    if not list_output:\n",
    "        all_preds = None\n",
    "        if return_softmax:\n",
    "            all_logits = None\n",
    "    else:\n",
    "        all_preds = [np.array([]) for _ in range(num_outputs)]\n",
    "\n",
    "    print('\\t\\tRunning prediction...')\n",
    "    # Get preds from model\n",
    "    for batch in eval_dataloader:\n",
    "\n",
    "        # Move to GPU\n",
    "        batch = tuple(t.to(device) for t in batch)\n",
    "\n",
    "        if span_mode:\n",
    "            input_ids, input_mask, labels, spans = batch\n",
    "        elif seg_mode:\n",
    "            input_ids, input_mask, labels, segment_ids = batch\n",
    "        else:\n",
    "            input_ids, input_mask, labels = batch\n",
    "\n",
    "        with torch.no_grad():\n",
    "            if span_mode:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=None,\n",
    "                            attention_mask=input_mask,\n",
    "                            spans=spans)\n",
    "            elif seg_mode:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=segment_ids,\n",
    "                            attention_mask=input_mask)\n",
    "            else:\n",
    "                out = model(input_ids,\n",
    "                            token_type_ids=None,\n",
    "                            attention_mask=input_mask)\n",
    "\n",
    "        label_ids = labels.to('cpu').numpy()\n",
    "        if all_labels is None:\n",
    "            all_labels = label_ids\n",
    "        else:\n",
    "            all_labels = np.concatenate((all_labels, label_ids), axis=0)\n",
    "        # print(all_labels.shape)\n",
    "\n",
    "        logits = out[0]\n",
    "        if list_output:  # This is broken, do not use\n",
    "            metr = {}\n",
    "            for o in range(num_outputs):\n",
    "                preds = torch.argmax(logits[o], dim=1).detach().cpu().numpy()\n",
    "                all_preds[o] = np.concatenate((all_preds[o], preds))\n",
    "                metr_o = compute_metrics(all_preds[o], all_labels, metrics)\n",
    "                for k, v in metr_o.items():\n",
    "                    metr[str(o) + '_' + k] = v\n",
    "        elif multilabel:\n",
    "            preds = torch.sigmoid(logits)\n",
    "            preds[preds > 0.5] = 1\n",
    "            preds[preds < 0.5] = 0\n",
    "            preds = preds.detach().cpu().numpy()\n",
    "            if all_preds is None:\n",
    "                all_preds = preds\n",
    "            else:\n",
    "                all_preds = np.concatenate((all_preds, preds), axis=0)\n",
    "\n",
    "            if return_softmax:\n",
    "                if not multilabel:\n",
    "                    logits = torch.softmax(logits,\n",
    "                                           dim=1).detach().cpu().numpy()\n",
    "                else:\n",
    "                    logits = torch.sigmoid(logits).detach().cpu().numpy()\n",
    "                if all_logits is None:\n",
    "                    all_logits = logits\n",
    "                else:\n",
    "                    all_logits = np.concatenate((all_logits, logits))\n",
    "\n",
    "        else:\n",
    "            preds = torch.argmax(\n",
    "                logits, dim=1 if not lm else 2).detach().cpu().numpy()\n",
    "            # print(preds.shape)\n",
    "            if all_preds is None:\n",
    "                all_preds = preds\n",
    "            else:\n",
    "                all_preds = np.concatenate((all_preds, preds))\n",
    "            # print(all_preds.shape)\n",
    "            if return_softmax:\n",
    "                logits = torch.softmax(\n",
    "                    logits, dim=1 if not lm else 2).detach().cpu().numpy()\n",
    "                if all_logits is None:\n",
    "                    all_logits = logits\n",
    "                else:\n",
    "                    all_logits = np.concatenate((all_logits, logits))\n",
    "\n",
    "    # Calculate metrics\n",
    "    print('\\t\\tComputing metrics...')\n",
    "    if multilabel:\n",
    "        # Overall metrics and per-label metrics\n",
    "        metr = compute_metrics(all_preds.flatten(), all_labels.flatten(),\n",
    "                               metrics)\n",
    "        for i in range(model.num_labels):\n",
    "            metr_i = compute_metrics(\n",
    "                all_preds.reshape(-1, model.num_labels)[:, i],\n",
    "                all_labels.reshape(-1, model.num_labels)[:, i].flatten(),\n",
    "                metrics)\n",
    "            for k in metr_i:\n",
    "                metr['%s_%s' % (str(k), str(i))] = metr_i[k]\n",
    "    elif lm:\n",
    "        # In language modeling, ignore examples where label is -100 and flatten\n",
    "        print('\\t\\t\\tFlattening and filtering preds and labels for LM...')\n",
    "        preds_temp = all_preds.flatten()\n",
    "        labels_temp = all_labels.flatten()\n",
    "        metr = compute_metrics(preds_temp[labels_temp != -100],\n",
    "                               labels_temp[labels_temp != -100], metrics)\n",
    "    else:\n",
    "        metr = compute_metrics(all_preds, all_labels, metrics)\n",
    "\n",
    "    print('\\tFinished evaluation in %ss.' % str(format_time(time.time() - t0)))\n",
    "\n",
    "    if not return_softmax:\n",
    "        return metr, all_preds, all_labels\n",
    "    else:\n",
    "        # Warning: this is not supported in list_output mode\n",
    "        return metr, all_preds, all_labels, all_logits\n",
    "\n",
    "\n",
    "def compute_metrics(preds, labels, metrics):\n",
    "    metr = {}\n",
    "    for m, m_name in metrics:\n",
    "        if m_name in ['accuracy', 'confusion_matrix']:\n",
    "            metr[m_name] = m(\n",
    "                labels, preds\n",
    "            )  # Assume each metric m will be a function of (y_true, y_pred)\n",
    "        else:\n",
    "            metr[m_name] = m(labels, preds, average='macro')\n",
    "    return metr\n",
    "\n",
    "\n",
    "# Save eval metrics (or any dictionary) to json file\n",
    "def save_results(results, output_dir, dataset_name):\n",
    "    with open(os.path.join(output_dir, 'results_%s.json' % str(dataset_name)),\n",
    "              'w') as f:\n",
    "        json.dump(results, f)\n",
    "\n",
    "\n",
    "# Print eval preds for a model on some dataset\n",
    "def save_preds(ids, labels, preds, output_dir, dataset_name):\n",
    "    assert len(ids) == len(labels) == len(preds)\n",
    "    if len(labels.shape) == 1:\n",
    "        with open(os.path.join(output_dir, 'preds_%s.tsv' % str(dataset_name)),\n",
    "                  'w') as f:\n",
    "            for exid, label, pred in zip(ids, labels, preds):\n",
    "                f.write(exid + '\\t' + str(int(label)) + '\\t' + str(int(pred)) +\n",
    "                        '\\n')\n",
    "    else:\n",
    "        with open(os.path.join(output_dir, 'preds_%s.tsv' % str(dataset_name)),\n",
    "                  'w') as f:\n",
    "            for exid, label, pred in zip(ids, labels, preds):\n",
    "                f.write(exid + '\\t' + '\\t'.join([str(int(l)) for l in label]) +\n",
    "                        '\\t' + '\\t'.join([str(int(p)) for p in pred]) + '\\n')\n",
    "\n",
    "\n",
    "# Print eval probs (softmax) for a model on some dataset\n",
    "def save_probs(ids, labels, preds, output_dir, dataset_name):\n",
    "    assert len(ids) == len(labels) == len(preds)\n",
    "    if len(labels.shape) == 1:\n",
    "        with open(os.path.join(output_dir, 'probs_%s.tsv' % str(dataset_name)),\n",
    "                  'w') as f:\n",
    "            for exid, label, pred in zip(ids, labels, preds):\n",
    "                f.write(exid + '\\t' + str(int(label)) + '\\t' +\n",
    "                        '\\t'.join([str(p) for p in pred]) + '\\n')\n",
    "    else:\n",
    "        with open(os.path.join(output_dir, 'probs_%s.tsv' % str(dataset_name)),\n",
    "                  'w') as f:\n",
    "            for exid, label, pred in zip(ids, labels, preds):\n",
    "                f.write(exid + '\\t' + '\\t'.join([str(int(l)) for l in label]) +\n",
    "                        '\\t' + '\\t'.join([str(p) for p in pred]) + '\\n')\n",
    "\n",
    "\n",
    "# Load model predictions (id, pred, label) from file\n",
    "def load_preds(fname):\n",
    "    lines = read_tsv(fname)\n",
    "    preds = {}\n",
    "    for l in lines:\n",
    "        exid, label, pred = l[0], int(float(l[1])), int(float(l[2]))\n",
    "        preds[exid] = {'label': label, 'pred': pred}\n",
    "    return preds\n",
    "\n",
    "\n",
    "# Get some generic metrics for a predicted and g.t. list of integers\n",
    "def list_comparison(pred, label):\n",
    "\n",
    "    if len(pred) > 0:\n",
    "        prec = len([p for p in pred if p in label]) / len(pred)\n",
    "    else:\n",
    "        prec = None\n",
    "\n",
    "    if len(label) > 0:\n",
    "        rec = len([p for p in pred if p in label]) / len(label)\n",
    "    else:\n",
    "        rec = None\n",
    "\n",
    "    # Define a \"correct\" prediction as a prediction with no incorrect values,\n",
    "    # which has at least one value if the label is non-empty\n",
    "    #\n",
    "    # Define a \"perfect\" prediction as an exact match of label and pred\n",
    "    if len(label) == 0:\n",
    "        if len(pred) == 0:\n",
    "            corr = True\n",
    "            perf = True\n",
    "        else:\n",
    "            corr = False\n",
    "            perf = False\n",
    "    else:\n",
    "        if len(pred) > 0 and len([p for p in pred if p not in label]) == 0:\n",
    "            corr = True\n",
    "        else:\n",
    "            corr = False\n",
    "\n",
    "        if len(pred) > 0 and set(pred) == set(label):\n",
    "            perf = True\n",
    "        else:\n",
    "            perf = False\n",
    "\n",
    "    return prec, rec, corr, perf\n",
    "\n",
    "\n",
    "# Run evaluation for the conflict detector\n",
    "def evaluate_tiered(model1,\n",
    "                    model2,\n",
    "                    eval_dataloader,\n",
    "                    device,\n",
    "                    metrics,\n",
    "                    seg_mode=False,\n",
    "                    return_softmax=False,\n",
    "                    return_explanations=False,\n",
    "                    return_losses=False,\n",
    "                    verbose=True):\n",
    "    if verbose:\n",
    "        print('\\tBeginning evaluation...')\n",
    "\n",
    "    t0 = time.time()\n",
    "\n",
    "    model1.zero_grad()\n",
    "    model2.zero_grad()\n",
    "    model1.eval()\n",
    "    model2.eval()\n",
    "    for layer in model1.precondition_classifiers:\n",
    "        layer.eval()\n",
    "    for layer in model.effect_classifiers:\n",
    "        layer1.eval()\n",
    "\n",
    "    all_pred_attributes = None\n",
    "    all_attributes = None\n",
    "\n",
    "    all_pred_prec = None\n",
    "    all_prec = None\n",
    "\n",
    "    all_pred_eff = None\n",
    "    all_eff = None\n",
    "\n",
    "    all_pred_conflicts = None\n",
    "    all_conflicts = None\n",
    "\n",
    "    all_pred_stories = None\n",
    "    all_stories = None\n",
    "    if return_softmax:\n",
    "        all_prob_stories = None\n",
    "\n",
    "    if verbose:\n",
    "        print('\\t\\tRunning prediction...')\n",
    "\n",
    "    if verbose:\n",
    "        bar_size = len(eval_dataloader)\n",
    "        bar = progressbar.ProgressBar(max_value=bar_size,\n",
    "                                      widgets=[\n",
    "                                          progressbar.Bar('#', '[', ']'), ' ',\n",
    "                                          progressbar.Percentage()\n",
    "                                      ])\n",
    "        bar_idx = 0\n",
    "        bar.start()\n",
    "\n",
    "    # Aggregate losses\n",
    "    agg_losses = {}\n",
    "\n",
    "    # Get preds from model\n",
    "    for batch in eval_dataloader:\n",
    "        # Move to GPU\n",
    "        batch = tuple(t.to(device) for t in batch)\n",
    "\n",
    "        input_ids = batch[0].long().to(device)\n",
    "        input_lengths = batch[1].to(device)\n",
    "        input_entities = batch[2].to(device)\n",
    "        input_mask = batch[3].to(device)\n",
    "        attributes = batch[4].long().to(device)\n",
    "        preconditions = batch[5].long().to(device)\n",
    "        effects = batch[6].long().to(device)\n",
    "        conflicts = batch[7].long().to(device)\n",
    "        labels = batch[8].long().to(device)\n",
    "        if seg_mode:\n",
    "            segment_ids = batch[9].to(device)\n",
    "        else:\n",
    "            segment_ids = None\n",
    "\n",
    "        batch_size, num_stories, num_entities, num_sents, seq_length = input_ids.shape\n",
    "\n",
    "        with torch.no_grad():\n",
    "            # out = model(input_ids,\n",
    "            #             input_lengths,\n",
    "            #             input_entities,\n",
    "            #             attention_mask=input_mask,\n",
    "            #             token_type_ids=segment_ids)\n",
    "            out_1 = model1(input_ids, \n",
    "                        input_lengths,\n",
    "                        input_entities,\n",
    "                        attention_mask=input_mask,\n",
    "                        token_type_ids=segment_ids,\n",
    "                        attributes=attributes,\n",
    "                        preconditions=preconditions,\n",
    "                        effects=effects,\n",
    "                        training=True)\n",
    "\n",
    "            out_preconditions_softmax=out_1['out_preconditions_softmax']\n",
    "            out_effects_softmax=out_1['out_effects_softmax']\n",
    "            outcls=out_1['out']\n",
    "            out_2 = model2(input_ids, \n",
    "                        input_lengths,\n",
    "                        input_entities,\n",
    "                        out=outcls,\n",
    "                        attention_mask=input_mask,\n",
    "                        token_type_ids=segment_ids,\n",
    "                        attributes=attributes,\n",
    "                        out_preconditions_softmax=out_preconditions_softmax,\n",
    "                        out_effects_softmax=out_effects_softmax,\n",
    "                        conflicts=conflicts,\n",
    "                        labels=labels,\n",
    "                        training=True)\n",
    "\n",
    "            out={}\n",
    "            for k in out_1:\n",
    "                out[k]=out_1[k]\n",
    "            for k in out_2:\n",
    "                out[k]=out_2[k]\n",
    "            out['total_loss']=ComputeLoss(out_1,out_2)\n",
    "\n",
    "            loss = out['total_loss']\n",
    "        if return_losses:\n",
    "            for k in out:\n",
    "                if 'loss' in k:\n",
    "                    if k not in agg_losses:\n",
    "                        agg_losses[k] = out[k]\n",
    "                    else:\n",
    "                        agg_losses[k] += out[k]\n",
    "\n",
    "        # Get gt/predicted attributes\n",
    "        if 'attributes' not in model.ablation:\n",
    "            label_ids = attributes.view(\n",
    "                -1, attributes.shape[-1]).to('cpu').numpy()\n",
    "            if all_attributes is None:\n",
    "                all_attributes = label_ids\n",
    "            else:\n",
    "                all_attributes = np.concatenate((all_attributes, label_ids),\n",
    "                                                axis=0)\n",
    "\n",
    "            preds = out['out_attributes'].detach().cpu().numpy()\n",
    "            preds[preds >= 0.5] = 1\n",
    "            preds[preds < 0.5] = 0\n",
    "            if all_pred_attributes is None:\n",
    "                all_pred_attributes = preds\n",
    "            else:\n",
    "                all_pred_attributes = np.concatenate(\n",
    "                    (all_pred_attributes, preds), axis=0)\n",
    "\n",
    "        # Get gt/predicted preconditions\n",
    "        label_ids = preconditions.view(\n",
    "            -1, preconditions.shape[-1]).to('cpu').numpy()\n",
    "        if all_prec is None:\n",
    "            all_prec = label_ids\n",
    "        else:\n",
    "            all_prec = np.concatenate((all_prec, label_ids), axis=0)\n",
    "\n",
    "        preds = out['out_preconditions'].detach().cpu().numpy()\n",
    "        if all_pred_prec is None:\n",
    "            all_pred_prec = preds\n",
    "        else:\n",
    "            all_pred_prec = np.concatenate((all_pred_prec, preds), axis=0)\n",
    "\n",
    "        # Get gt/predicted preconditions\n",
    "        label_ids = effects.view(-1, effects.shape[-1]).to('cpu').numpy()\n",
    "        if all_eff is None:\n",
    "            all_eff = label_ids\n",
    "        else:\n",
    "            all_eff = np.concatenate((all_eff, label_ids), axis=0)\n",
    "\n",
    "        preds = out['out_effects'].detach().cpu().numpy()\n",
    "        if all_pred_eff is None:\n",
    "            all_pred_eff = preds\n",
    "        else:\n",
    "            all_pred_eff = np.concatenate((all_pred_eff, preds), axis=0)\n",
    "\n",
    "        # Get gt/predicted conflict points\n",
    "        label_ids = conflicts.to('cpu').numpy()\n",
    "        if all_conflicts is None:\n",
    "            all_conflicts = label_ids\n",
    "        else:\n",
    "            all_conflicts = np.concatenate((all_conflicts, label_ids), axis=0)\n",
    "\n",
    "        # preds_start = torch.argmax(out['out_start'],dim=-1).detach().cpu().numpy()\n",
    "        # preds_end = torch.argmax(out['out_end'],dim=-1).detach().cpu().numpy()\n",
    "        # preds = np.stack((preds_start, preds_end), axis=1)\n",
    "\n",
    "        preds = out['out_conflicts'].detach().cpu().numpy()\n",
    "        preds[preds < 0.5] = 0.\n",
    "        preds[preds >= 0.5] = 1.\n",
    "        if all_pred_conflicts is None:\n",
    "            all_pred_conflicts = preds\n",
    "        else:\n",
    "            all_pred_conflicts = np.concatenate((all_pred_conflicts, preds),\n",
    "                                                axis=0)\n",
    "\n",
    "        # Get gt/predicted story choices\n",
    "        label_ids = labels.to('cpu').numpy()\n",
    "        if all_stories is None:\n",
    "            all_stories = label_ids\n",
    "        else:\n",
    "            all_stories = np.concatenate((all_stories, label_ids), axis=0)\n",
    "\n",
    "        preds = torch.argmax(out['out_stories'], dim=-1).detach().cpu().numpy()\n",
    "        if all_pred_stories is None:\n",
    "            all_pred_stories = preds\n",
    "        else:\n",
    "            all_pred_stories = np.concatenate((all_pred_stories, preds),\n",
    "                                              axis=0)\n",
    "        if return_softmax:\n",
    "            probs = torch.softmax(out['out_stories'],\n",
    "                                  dim=-1).detach().cpu().numpy()\n",
    "            if all_prob_stories is None:\n",
    "                all_prob_stories = probs\n",
    "            else:\n",
    "                all_prob_stories = np.concatenate((all_prob_stories, probs),\n",
    "                                                  axis=0)\n",
    "\n",
    "        if verbose:\n",
    "            bar_idx += 1\n",
    "            bar.update(bar_idx)\n",
    "    if verbose:\n",
    "        bar.finish()\n",
    "\n",
    "    # Calculate metrics\n",
    "    if verbose:\n",
    "        print('\\t\\tComputing metrics...')\n",
    "\n",
    "    # print(all_pred_attributes.shape)\n",
    "    # print(all_attributes.shape)\n",
    "    # print(all_pred_prec.shape)\n",
    "    # print(all_prec.shape)\n",
    "    # print(all_pred_eff.shape)\n",
    "    # print(all_eff.shape)\n",
    "    # print(all_pred_conflicts.shape)\n",
    "    # print(all_conflicts.shape)\n",
    "    # print(all_pred_stories.shape)\n",
    "    # print(all_stories.shape)\n",
    "\n",
    "    input_lengths = input_lengths.detach().cpu().numpy()\n",
    "\n",
    "    # Overall metrics and per-category metrics for attributes, preconditions, and effects\n",
    "    # NOTE: there are a lot of extra negative examples due to padding along sentene and entity dimenions. This can't affect F1, but will affect accuracy and make it disproportionately large.\n",
    "    metr_attr = None\n",
    "    if 'attributes' not in model.ablation:\n",
    "        metr_attr = compute_metrics(all_pred_attributes.flatten(),\n",
    "                                    all_attributes.flatten(), metrics)\n",
    "        for i in range(model.num_attributes):\n",
    "            metr_i = compute_metrics(all_pred_attributes[:, i],\n",
    "                                     all_attributes[:, i], metrics)\n",
    "            for k in metr_i:\n",
    "                metr_attr['%s_%s' % (str(k), str(i))] = metr_i[k]\n",
    "\n",
    "    metr_prec = compute_metrics(all_pred_prec.flatten(), all_prec.flatten(),\n",
    "                                metrics)\n",
    "    for i in range(model.num_attributes):\n",
    "        metr_i = compute_metrics(all_pred_prec[:, i], all_prec[:, i], metrics)\n",
    "        for k in metr_i:\n",
    "            metr_prec['%s_%s' % (str(k), str(i))] = metr_i[k]\n",
    "\n",
    "    metr_eff = compute_metrics(all_pred_eff.flatten(), all_eff.flatten(),\n",
    "                               metrics)\n",
    "    for i in range(model.num_attributes):\n",
    "        metr_i = compute_metrics(all_pred_eff[:, i], all_eff[:, i], metrics)\n",
    "        for k in metr_i:\n",
    "            metr_eff['%s_%s' % (str(k), str(i))] = metr_i[k]\n",
    "\n",
    "    # Conflict span metrics\n",
    "    metr_conflicts = compute_metrics(all_pred_conflicts.flatten(),\n",
    "                                     all_conflicts.flatten(), metrics)\n",
    "\n",
    "    # metr_start = compute_metrics(all_pred_spans[:,0], all_spans[:,0], metrics)\n",
    "    # for k in metr_start:\n",
    "    #   metr[k + '_start'] = metr_start[k]\n",
    "\n",
    "    # metr_end = compute_metrics(all_pred_spans[:,1], all_spans[:,1], metrics)\n",
    "    # for k in metr_end:\n",
    "    #   metr[k + '_end'] = metr_end[k]\n",
    "\n",
    "    metr_stories = compute_metrics(all_pred_stories.flatten(),\n",
    "                                   all_stories.flatten(), metrics)\n",
    "\n",
    "    verifiability, explanations = verifiable_reasoning(\n",
    "        all_stories,\n",
    "        all_pred_stories,\n",
    "        all_conflicts,\n",
    "        all_pred_conflicts,\n",
    "        all_prec,\n",
    "        all_pred_prec,\n",
    "        all_eff,\n",
    "        all_pred_eff,\n",
    "        return_explanations=True)\n",
    "    metr_stories['verifiability'] = verifiability\n",
    "\n",
    "    if verbose:\n",
    "        print('\\tFinished evaluation in %ss.' %\n",
    "              str(format_time(time.time() - t0)))\n",
    "\n",
    "    return_base = [\n",
    "        metr_attr, all_pred_attributes, all_attributes, metr_prec,\n",
    "        all_pred_prec, all_prec, metr_eff, all_pred_eff, all_eff,\n",
    "        metr_conflicts, all_pred_conflicts, all_conflicts, metr_stories,\n",
    "        all_pred_stories, all_stories\n",
    "    ]\n",
    "    if return_softmax:\n",
    "        return_base += [all_prob_stories]\n",
    "    if return_explanations:\n",
    "        return_base += [explanations]\n",
    "    if return_losses:\n",
    "        for k in agg_losses:\n",
    "            if 'loss' in k:\n",
    "                agg_losses[k] /= len(eval_dataloader)\n",
    "        return_base += [agg_losses]\n",
    "\n",
    "    return tuple(return_base)\n",
    "\n",
    "\n",
    "# \"Verifiability\" metric: % of examples where\n",
    "# 1) Story prediction is correct\n",
    "# 2) Conflicting sentences are correct\n",
    "# 3) All nontrivial predicted states in the conflicting sentences are correct\n",
    "def verifiable_reasoning(stories,\n",
    "                         pred_stories,\n",
    "                         conflicts,\n",
    "                         pred_conflicts,\n",
    "                         preconditions,\n",
    "                         pred_preconditions,\n",
    "                         effects,\n",
    "                         pred_effects,\n",
    "                         return_explanations=False):\n",
    "    atts = list(att_default_values.keys())\n",
    "\n",
    "    verifiable = 0\n",
    "    total = 0\n",
    "    explanations = []\n",
    "    for i, ex in enumerate(stories):\n",
    "        l_story = stories[i]\n",
    "        p_story = pred_stories[i]\n",
    "\n",
    "        l_conflict = np.sum(conflicts, axis=(1, 2))[i]\n",
    "        p_conflict = np.sum(pred_conflicts.reshape(conflicts.shape),\n",
    "                            axis=(1, 2))[i]\n",
    "        l_conflict = np.nonzero(l_conflict)[0]\n",
    "        p_conflict = np.nonzero(p_conflict)[0]\n",
    "\n",
    "        l_prec = preconditions.reshape(\n",
    "            list(conflicts.shape[:4]) + [preconditions.shape[-1]])[\n",
    "                i,\n",
    "                1 - l_story]  # (num entities, num sentences, num attributes)\n",
    "        p_prec = pred_preconditions.reshape(\n",
    "            list(conflicts.shape[:4]) + [preconditions.shape[-1]])[\n",
    "                i,\n",
    "                1 - l_story]  # (num entities, num sentences, num attributes)\n",
    "\n",
    "        l_eff = effects.reshape(\n",
    "            list(conflicts.shape[:4]) + [effects.shape[-1]])[\n",
    "                i,\n",
    "                1 - l_story]  # (num entities, num sentences, num attributes)\n",
    "        p_eff = pred_effects.reshape(\n",
    "            list(conflicts.shape[:4]) + [effects.shape[-1]])[\n",
    "                i,\n",
    "                1 - l_story]  # (num entities, num sentences, num attributes)\n",
    "\n",
    "        explanation = {\n",
    "            'story_label': int(l_story),\n",
    "            'story_pred': int(p_story),\n",
    "            'conflict_label': [int(c) for c in l_conflict],\n",
    "            'conflict_pred': [int(c) for c in p_conflict],\n",
    "            'preconditions_label': l_prec,\n",
    "            'preconditions_pred': p_prec,\n",
    "            'effects_label': l_eff,\n",
    "            'effects_pred': p_eff,\n",
    "            'valid_explanation': False\n",
    "        }\n",
    "\n",
    "        if l_story == p_story:\n",
    "            if len(l_conflict) == len(p_conflict) == 2:\n",
    "                if l_conflict[0] == p_conflict[0] and l_conflict[\n",
    "                        1] == p_conflict[1]:\n",
    "                    states_verifiable = True\n",
    "                    found_states = False\n",
    "\n",
    "                    # Check that effect of first conflict sentence has states which are correct\n",
    "                    for sl, sp in [(l_eff, p_eff)\n",
    "                                   ]:  # Check preconditions and effects\n",
    "                        for sl_e, sp_e in zip(sl, sp):  # Check all entities\n",
    "                            for si in [l_conflict[0]\n",
    "                                       ]:  # Check conflicting sentences\n",
    "                                sl_es = sl_e[si]\n",
    "                                sp_es = sp_e[si]\n",
    "                                for j, p in enumerate(\n",
    "                                        sp_es\n",
    "                                ):  # Check all attributes where there's a nontrivial prediction\n",
    "                                    if p != att_default_values[atts[\n",
    "                                            j]] and p > 0:  # NOTE: p > 0 is required to avoid counting any padding predictions.\n",
    "                                        found_states = True\n",
    "                                        if p != sl_es[j]:\n",
    "                                            states_verifiable = False\n",
    "\n",
    "                    # Check that precondition of second conflict sentence has states which are correct\n",
    "                    for sl, sp in [(l_prec, p_prec)\n",
    "                                   ]:  # Check preconditions and effects\n",
    "                        for sl_e, sp_e in zip(sl, sp):  # Check all entities\n",
    "                            for si in [l_conflict[1]\n",
    "                                       ]:  # Check conflicting sentences\n",
    "                                sl_es = sl_e[si]\n",
    "                                sp_es = sp_e[si]\n",
    "                                for j, p in enumerate(\n",
    "                                        sp_es\n",
    "                                ):  # Check all attributes where there's a nontrivial prediction\n",
    "                                    if p != att_default_values[atts[\n",
    "                                            j]] and p > 0:  # NOTE: p > 0 is required to avoid counting any padding predictions.\n",
    "                                        found_states = True\n",
    "                                        if p != sl_es[j]:\n",
    "                                            states_verifiable = False\n",
    "\n",
    "                    if states_verifiable and found_states:\n",
    "                        verifiable += 1\n",
    "                        explanation['valid_explanation'] = True\n",
    "\n",
    "        total += 1\n",
    "        explanations.append(explanation)\n",
    "\n",
    "    if not return_explanations:\n",
    "        return verifiable / total\n",
    "    else:\n",
    "        return verifiable / total, explanations\n",
    "\n",
    "\n",
    "# Adds entity and attribute labels to explanations object returned from eval function (for easier to read physical states)\n",
    "def add_entity_attribute_labels(explanations, dataset, attributes):\n",
    "    for x, expl in enumerate(explanations):\n",
    "        ex = dataset[x]\n",
    "        bad_story = ex['stories'][1 - ex['label']]\n",
    "        expl['example_id'] = ex['example_id']\n",
    "        expl['story0'] = '\\n'.join(ex['stories'][0]['sentences'])\n",
    "        expl['story1'] = '\\n'.join(ex['stories'][1]['sentences'])\n",
    "        assert ex['label'] == expl[\n",
    "            'story_label'], \"mismatch between explanations and original examples!\"\n",
    "\n",
    "        entities = [d['entity'] for d in bad_story['entities']]\n",
    "        for key in [\n",
    "                'preconditions_label', 'preconditions_pred', 'effects_label',\n",
    "                'effects_pred'\n",
    "        ]:\n",
    "            new_states = {}\n",
    "            for i, ent_anns in enumerate(expl[key]):\n",
    "                if i < len(entities):\n",
    "                    ent = entities[i]\n",
    "                    new_states[ent] = {}\n",
    "                    for j, sent_anns in enumerate(ent_anns):\n",
    "                        if j < len(bad_story['sentences']):\n",
    "                            new_states[ent][j] = {}\n",
    "                            for k, att_ann in enumerate(sent_anns):\n",
    "                                if int(att_ann) != att_default_values[\n",
    "                                        attributes[k]] and int(att_ann) > 0:\n",
    "                                    att = attributes[k]\n",
    "                                    new_states[ent][j][att] = int(att_ann)\n",
    "            expl[key] = new_states\n",
    "        explanations[x] = expl\n",
    "    return explanations"
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