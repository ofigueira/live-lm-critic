import sys
import torch
import random
import hashlib
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import OPTForCausalLM, GPTNeoForCausalLM
from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig
from transformers import XLMRobertaTokenizer, XLMRobertaForCausalLM, XLMRobertaConfig
from transformers import BartTokenizer, BartForCausalLM
import nltk
import pandas as pd
nltk.download('punkt')

sys.path.insert(0, '.')
from critic.perturbations import get_local_neighbors_char_level, get_local_neighbors_word_level
from utils.spacy_tokenizer import spacy_tokenize_gec

def get_gpt2_loss(model, tokenizer, input_ids, attention_mask, labels):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_logits = outputs[1] #[bsize, seqlen, vocab]
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bsize, seqlen-1)
            loss = (loss * shift_mask).sum(dim=1) #[bsize, ]
        return loss


MAX_LENGTH = 66

def run_gpt2(sents, model, tokenizer, cuda=False, model_name=None):
    assert isinstance(sents, list)
    _sents = [tokenizer.bos_token + s for s in sents]
    inputs = tokenizer(_sents, return_tensors="pt", padding=True)
    if inputs['input_ids'].size(1) > MAX_LENGTH:
        return None
    if cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    loss = get_gpt2_loss(model, tokenizer, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
    logps = - loss.detach().cpu()
    return logps


def gpt2_critic_char_level_only(sent, verbose=1, cuda=False, fp16=True, seed='auto', n_samples=100):
    return_string = []
    if seed == 'auto':
        seed = int(hashlib.md5(sent.encode()).hexdigest(), 16) % (2**32) #Seed must be between 0 and 2**32 - 1
    if verbose > 1:
        print ('seed', seed)
    np.random.seed(seed); random.seed(seed)
    is_good = True
    for _ in range(1):
        sent_perturbations = get_local_neighbors_char_level(sent, max_n_samples=n_samples)
        if verbose > 1:
            print ("#sent_perturbations (char-level)", len(sent_perturbations))
            return_string.append(f"#sent_perturbations (char-level){len(sent_perturbations)}\n")
        sents = [sent] + list(sent_perturbations)
        if fp16:
            with torch.cuda.amp.autocast():
                logps = run_gpt2(sents, cuda)
        else:
            logps = run_gpt2(sents, cuda)
        if logps is None:
            if verbose:
                print ('Invalid input. Maybe the sentence is too long.')
                return_string.append('Invalid input. Maybe the sentence is too long.\n')
            return None
        best_idx = int(logps.argmax())
        if best_idx != 0:
            is_good = False
            break
    if verbose:
        if is_good:
            print ('Good! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
            return_string.append('Good! Your sentence log(p) = {:.3f}\n'.format(float(logps[0])))
        else:
            print ('Bad! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
            return_string.append('Bad! Your sentence log(p) = {:.3f}\n'.format(float(logps[0])))
            print ('Neighbor sentence with highest log(p): {} (= {:.3f})'.format(sents[best_idx], float(logps[best_idx])))
            return_string.append('Neighbor sentence with highest log(p): {} (= {:.3f})\n'.format(sents[best_idx], float(logps[best_idx])))
    counter_example = None
    if not is_good:
        counter_example = [sents[best_idx], float(logps[best_idx])]
    return is_good, float(logps[0]), counter_example


def gpt2_critic(sent, model, tokenizer, verbose=1, cuda=False, fp16=True, seed='auto', n_samples=100, word_level_mode='refine'):
    return_string = []
    if seed == 'auto':
        seed = int(hashlib.md5(sent.encode()).hexdigest(), 16) % (2**32) #Seed must be between 0 and 2**32 - 1
    if verbose > 1:
        print ('seed', seed)
        return_string.append(f'seed{seed}\n')
    np.random.seed(seed); random.seed(seed)
    sent_toked = spacy_tokenize_gec(sent)
    is_good = True
    for _ in range(1):
        sent_perturbations_w, orig_sent = get_local_neighbors_word_level(sent_toked, max_n_samples=n_samples//2, mode=word_level_mode)
        sent_perturbations_c = get_local_neighbors_char_level(orig_sent, max_n_samples=n_samples//2)
        if verbose > 1:
            print ("#sent_perturbations (char-level)", len(sent_perturbations_c))
            return_string.append("#sent_perturbations (char-level)\n", len(sent_perturbations_c))
            print ("#sent_perturbations (word-level)", len(sent_perturbations_w))
            return_string.append("#sent_perturbations (word-level)\n", len(sent_perturbations_w))
        sents = [orig_sent] + list(sent_perturbations_c.union(sent_perturbations_w))
        if fp16:
            with torch.cuda.amp.autocast():
                logps = run_gpt2(sents, model, tokenizer, cuda)
        else:
            logps = run_gpt2(sents, model, tokenizer, cuda)
        if logps is None:
            if verbose:
                print ('Invalid input. Maybe the sentence is too long.')
                return_string.append('Invalid input. Maybe the sentence is too long.\n')
            return None
        best_idx = int(logps.argmax())
        if best_idx != 0:
            is_good = False
            break
    if verbose:
        if is_good:
            print ('Good! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
            return_string.append('Good! Your sentence log(p) = {:.3f}\n'.format(float(logps[0])))
        else:
            print ('Bad! Your sentence log(p) = {:.3f}'.format(float(logps[0])))
            return_string.append('Bad! Your sentence log(p) = {:.3f}\n'.format(float(logps[0])))
            print ('Neighbor sentence with highest log(p): {} (= {:.3f})'.format(sents[best_idx], float(logps[best_idx])))
            return_string.append('Neighbor sentence with highest log(p): {} (= {:.3f})\n'.format(sents[best_idx], float(logps[best_idx])))
    counter_example = None
    if not is_good:
        counter_example = [sents[best_idx], float(logps[best_idx])]
    return is_good, float(logps[0]), counter_example, return_string


def main():
    import streamlit as st
    st.subheader('Exploring Unsupervised Grammatical Error Correction with Transformer-Based Models')
    sent = st.text_input('Enter a sentence:', value="")

    ### LMs we are trying:
    if sent != '':
        st.markdown(f"**Input Sentence**: {sent}")
        results = {}

        with st.spinner('Running with GPT-2 LM...'):
            ## GPT-2 LM (original LM-critic)
            model_name = 'gpt2'
            nice_name = "GPT-2"
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.eval()
            model.cpu()
            is_good, score, counter_example, return_string_GPT2 = gpt2_critic(sent, model, tokenizer)
        st.markdown("**Results with GPT-2 LM:**")
        st.write('\n'.join(return_string_GPT2))
        results[nice_name] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with OPT LM...'):
            ## OPT LM
            model_name = "facebook/opt-350m"
            nice_name = "OPT"
            model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
            tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.cpu()
            is_good, score, counter_example, return_string_OPT = gpt2_critic(sent, model, tokenizer)
        st.markdown("**Results with OPT LM:**")
        st.write('\n'.join(return_string_OPT))
        results[nice_name] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with GPT NEO LM...'):
            ## GPT NEO
            model_name = "EleutherAI/gpt-neo-1.3B"
            nice_name = "GPT NEO"
            model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
            tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.cpu()
            is_good, score, counter_example, return_string_GPTNEO = gpt2_critic(sent, model, tokenizer)
        st.markdown("**Results with GPT NEO LM:**")
        st.write('\n'.join(return_string_GPTNEO))
        results[nice_name] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with RoBERTa LM...'):
            ## RoBERTa
            model_name = "roberta-base"
            nice_name = "RoBERTa"
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            config = RobertaConfig.from_pretrained("roberta-base")
            config.is_decoder = True
            model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.cpu()
            is_good, score, counter_example, return_string_RoBERTa = gpt2_critic(sent, model, tokenizer)
        st.markdown("**Results with RoBERTa LM:**")
        st.write('\n'.join(return_string_RoBERTa))
        results[nice_name] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with BART LM...'):
            ## RoBERTa
            model_name = "facebook/bart-base"
            nice_name = "BART"
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
            assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.cpu()
            is_good, score, counter_example, return_string_BART = gpt2_critic(sent, model, tokenizer)
        st.markdown("**Results with BART LM:**")
        st.write('\n'.join(return_string_BART))
        results[nice_name] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with XLM RoBERTa LM...'):
            ## XLM RoBERTa
            model_name = 'xlm-roberta-base'
            nice_name = 'XLM RoBERTa'
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")
            config.is_decoder = True
            model = XLMRobertaForCausalLM.from_pretrained("xlm-roberta-base", config=config)
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.cpu()
            is_good, score, counter_example, return_string_XLMRoBERTa = gpt2_critic(sent, model, tokenizer)
        st.markdown("**Results with XLM RoBERTa LM:**")
        st.write('\n'.join(return_string_XLMRoBERTa))
        results[nice_name] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        df = pd.DataFrame.from_dict(results, 
            orient = 'index',
            columns=['Judgement', 'Score (log(p))', 'Neighbor sentence with highest score (log(p))', 'Neighbor sentence score (log(p))'])
        st.markdown("**Tabular summary of results:**")
        st.table(df)

        st.write("Done.")

if __name__ == '__main__':
    main()
