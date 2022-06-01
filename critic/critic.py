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
import streamlit as st

st.subheader('Exploring Unsupervised Grammatical Error Correction with Transformer-Based Models')
st.write('This live demonstration is adapted from the paper [LM-Critic: Language Models for Unsupervised Grammatical Error Correction](https://aclanthology.org/2021.emnlp-main.611.pdf) (EMNLP 2021) by Michihiro Yasunaga, Jure Leskovec, Percy Liang.')
st.write('The below demo first loads several LMs that we use in the LM-Critic. You will be prompted to enter a sentence which will then be scored by each of the LM-Critics using different LMs.')

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

@st.cache(suppress_st_warning=True)
def init_lms():
    placeholder_lm_name = st.empty()
    my_bar = st.progress(10)
    prog = 10

    ## GPT-2 LM (original LM-critic)
    model_name_gpt2 = 'gpt2'
    nice_name_gpt2 = "GPT-2"
    placeholder_lm_name.text(f"Initializing {nice_name_gpt2}...")
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(model_name_gpt2)
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
    model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name_gpt2)
    model_gpt2.eval()
    model_gpt2.cpu()
    st.session_state["model_gpt2"] = model_gpt2
    st.session_state["tokenizer_gpt2"] = tokenizer_gpt2
    st.session_state["nice_name_gpt2"] = nice_name_gpt2

    prog += 10
    my_bar.progress(prog)    

    ## OPT LM
    model_name_opt = "facebook/opt-350m"
    nice_name_opt = "OPT"
    placeholder_lm_name.text(f"Initializing {nice_name_opt}...")
    model_opt = OPTForCausalLM.from_pretrained(model_name_opt)
    tokenizer_opt = GPT2Tokenizer.from_pretrained(model_name_opt)
    tokenizer_opt.pad_token = tokenizer_opt.eos_token
    model_opt.eval()
    model_opt.cpu()
    st.session_state["model_opt"] = model_opt
    st.session_state["tokenizer_opt"] = tokenizer_opt
    st.session_state["nice_name_opt"] = nice_name_opt

    prog += 10
    my_bar.progress(prog)  

    ## GPT NEO
    model_name_gptneo = "EleutherAI/gpt-neo-1.3B"
    nice_name_gptneo = "GPT NEO"
    placeholder_lm_name.text(f"Initializing {nice_name_gptneo}...")
    model_gptneo = GPTNeoForCausalLM.from_pretrained(model_name_gptneo)
    tokenizer_gptneo = GPT2Tokenizer.from_pretrained(model_name_gptneo)
    tokenizer_gptneo.pad_token = tokenizer_gptneo.eos_token
    model_gptneo.eval()
    model_gptneo.cpu()
    st.session_state["model_gptneo"] = model_gptneo
    st.session_state["tokenizer_gptneo"] = tokenizer_gptneo
    st.session_state["nice_name_gptneo"] = nice_name_gptneo

    prog += 10
    my_bar.progress(prog)  

    ## RoBERTa
    model_name_roberta = "roberta-base"
    nice_name_roberta = "RoBERTa"
    placeholder_lm_name.text(f"Initializing {nice_name_roberta}...")
    tokenizer_roberta = RobertaTokenizer.from_pretrained(model_name_roberta)
    config_roberta = RobertaConfig.from_pretrained(model_name_roberta)
    config_roberta.is_decoder = True
    model_roberta = RobertaForCausalLM.from_pretrained(model_name_roberta, config=config_roberta)
    tokenizer_roberta.pad_token = tokenizer_roberta.eos_token
    model_roberta.eval()
    model_roberta.cpu()
    st.session_state["model_roberta"] = model_gptneo
    st.session_state["tokenizer_roberta"] = tokenizer_roberta
    st.session_state["nice_name_roberta"] = nice_name_roberta

    prog += 10
    my_bar.progress(prog)  

    ## BART
    model_name_bart = "facebook/bart-base"
    nice_name_bart = "BART"
    placeholder_lm_name.text(f"Initializing {nice_name_bart}...")
    tokenizer_bart = BartTokenizer.from_pretrained(model_name_bart)
    model_bart = BartForCausalLM.from_pretrained(model_name_bart, add_cross_attention=False)
    assert model_bart.config.is_decoder, f"{model_bart.__class__} has to be configured as a decoder."
    tokenizer_bart.pad_token = tokenizer_bart.eos_token
    model_bart.eval()
    model_bart.cpu()
    st.session_state["model_bart"] = model_bart
    st.session_state["tokenizer_bart"] = tokenizer_bart
    st.session_state["nice_name_bart"] = nice_name_bart

    prog += 10
    my_bar.progress(prog)  

    ## XLM RoBERTa
    model_name_xlmroberta = 'xlm-roberta-base'
    nice_name_xlmroberta = 'XLM RoBERTa'
    placeholder_lm_name.text(f"Initializing {nice_name_xlmroberta}...")
    tokenizer_xlmroberta = XLMRobertaTokenizer.from_pretrained(model_name_xlmroberta)
    config_xlmroberta = XLMRobertaConfig.from_pretrained(model_name_xlmroberta)
    config_xlmroberta.is_decoder = True
    model_xlmroberta = XLMRobertaForCausalLM.from_pretrained(model_name_xlmroberta, config=config_xlmroberta)
    tokenizer_xlmroberta.pad_token = tokenizer_xlmroberta.eos_token
    model_xlmroberta.eval()
    model_xlmroberta.cpu()
    st.session_state["model_xlmroberta"] = model_xlmroberta
    st.session_state["tokenizer_xlmroberta"] = tokenizer_xlmroberta
    st.session_state["nice_name_xlmroberta"] = nice_name_xlmroberta

    prog += 10
    my_bar.progress(prog)
    placeholder_lm_name.empty()
    my_bar.empty()

def main():
    init_lms()
    sent = st.text_input('Enter a sentence:', value="")

    ### LMs we are trying:
    if sent != '':
        st.markdown(f"**Input Sentence**: {sent}")
        results = {}

        with st.spinner('Running with GPT-2 LM...'):
            ## GPT-2 LM (original LM-critic)
            is_good, score, counter_example, return_string_GPT2 = gpt2_critic(sent, st.session_state['model_gpt2'], st.session_state['tokenizer_gpt2'])
        st.markdown("**Results with GPT-2 LM:**")
        st.write('\n'.join(return_string_GPT2))
        results[st.session_state['nice_name_gpt2']] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with OPT LM...'):
            ## OPT LM
            is_good, score, counter_example, return_string_OPT = gpt2_critic(sent, st.session_state['model_opt'], st.session_state['tokenizer_opt'])
        st.markdown("**Results with OPT LM:**")
        st.write('\n'.join(return_string_OPT))
        results[st.session_state['nice_name_opt']] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with GPT NEO LM...'):
            ## GPT NEO
            is_good, score, counter_example, return_string_GPTNEO = gpt2_critic(sent, st.session_state['model_gptneo'], st.session_state['tokenizer_gptneo'])
        st.markdown("**Results with GPT NEO LM:**")
        st.write('\n'.join(return_string_GPTNEO))
        results[st.session_state['nice_name_gptneo']] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with RoBERTa LM...'):
            ## RoBERTa
            is_good, score, counter_example, return_string_RoBERTa = gpt2_critic(sent, st.session_state['model_roberta'], st.session_state['tokenizer_roberta'])
        st.markdown("**Results with RoBERTa LM:**")
        st.write('\n'.join(return_string_RoBERTa))
        results[st.session_state['nice_name_roberta']] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with BART LM...'):
            ## BART
            is_good, score, counter_example, return_string_BART = gpt2_critic(sent, st.session_state['model_bart'], st.session_state['tokenizer_bart'])
        st.markdown("**Results with BART LM:**")
        st.write('\n'.join(return_string_BART))
        results[st.session_state['nice_name_bart']] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        with st.spinner('Running with XLM RoBERTa LM...'):
            ## XLM RoBERTa
            is_good, score, counter_example, return_string_XLMRoBERTa = gpt2_critic(sent, st.session_state['model_xlmroberta'], st.session_state['tokenizer_xlmroberta'])
        st.markdown("**Results with XLM RoBERTa LM:**")
        st.write('\n'.join(return_string_XLMRoBERTa))
        results[st.session_state['nice_name_xlmroberta']] = ["Good" if is_good else "Bad", str(round(score, 3)), "N/A" if not counter_example else str(counter_example[0]), "N/A" if not counter_example else str(round(counter_example[1], 3))]

        df = pd.DataFrame.from_dict(results, 
            orient = 'index',
            columns=['Judgement', 'Score (log(p))', 'Neighbor sentence with highest score (log(p))', 'Neighbor sentence score (log(p))'])
        st.markdown("**Tabular summary of results:**")
        st.table(df)

        st.write("Input another sentence!")

if __name__ == '__main__':
    main()
