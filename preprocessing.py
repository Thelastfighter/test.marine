import numpy as np
from hparams import hparams
import librosa
import glob
import os
# from collections import defaultdict
def wav2feature(wav_file,para):
    wav,_ = librosa.load(wav_file,sr = None,mono = True)
    fbank = librosa.feature.melspectrogram(wav,
                                           sr =para.fs,
                                           n_fft = para.n_fft,
                                           win_length = para.win_length,
                                           hop_length = para.hop_length,
                                           n_mels=para.n_mels,
                                           fmin=para.fmin, 
                                           fmax=para.fmax)
    log_fbank = librosa.power_to_db(fbank, ref=np.max)
    return log_fbank
    
    
def processing_wavs(wav_files,para):
    feas = []
    ids = []
    for file in wav_files:
        print("processing file %s"%(file))
        id_wav = os.path.split(file)[-1][:-4] 
        fea = wav2feature(file,para)
        feas.append(fea)
        ids.append(id_wav)
    
    # 计算特征的 均值和方差
    fea_array = np.concatenate(feas,axis=1)  # fea的维度 D *T
    fea_mean =  np.mean(fea_array,axis=1,keepdims = True)
    fea_std =  np.std(fea_array,axis=1,keepdims = True)
    
    save_path = para.path_fea
    os.makedirs(save_path,exist_ok = True)
    
    # 对所有的特征进行正则, 并保存
    for fea,id_wav in zip(feas,ids):
        norm_fea = (fea- fea_mean)/ fea_std
        fea_name = os.path.join(save_path,id_wav+'.npy')  
        np.save(fea_name,norm_fea)
    
    static_name = os.path.join(save_path,'static.npy')
    np.save(static_name,np.array([fea_mean,fea_std],dtype=object))
    
'''
文本处理部分
'''

'''
文本处理部分
'''
# ascii code, used to delete Chinese punctuation
CHN_PUNC_LIST = [183, 215, 8212, 8216, 8217, 8220, 8221, 8230,
    12289, 12290, 12298, 12299, 12302, 12303, 12304, 12305,
    65281, 65288, 65289, 65292, 65306, 65307, 65311]
CHN_PUNC_SET = set(CHN_PUNC_LIST)

MANDARIN_INITIAL_LIST = ["b", "ch", "c", "d", "f", "g", "h", "j",\
    "k", "l", "m", "n", "p", "q", "r", "sh", "s", "t", "x", "zh", "z"]

# prosody phone list
CHN_PHONE_PUNC_LIST = ['sp2', 'sp1', 'sil']
# erhua phoneme
CODE_ERX = 0x513F

def _update_insert_pos(old_pos, pylist):
    new_pos = old_pos + 1
    i = new_pos
    while i < len(pylist)-1:
        # if the first letter is upper, then this is the phoneme of English letter
        if pylist[i][0].isupper():
            i += 1
            new_pos += 1
        else:
            break
    return new_pos

def _pinyin_preprocess(line, words):
    if line.find('.') >= 0:
        # remove '.' in English letter phonemes, for example: 'EH1 F . EY1 CH . P IY1'
        py_list = line.replace('/', '').strip().split('.')
        py_str = ''.join(py_list)
        pinyin = py_str.split()
    else:
        pinyin = line.replace('/', '').strip().split()

    # now the content in pinyin like: ['OW1', 'K', 'Y', 'UW1', 'JH', 'EY1', 'shi4', 'yi2', 'ge4']
    insert_pos = _update_insert_pos(-1, pinyin)
    i = 0
    while i < len(words):
        if ord(words[i]) in CHN_PUNC_SET:
            i += 1
            continue
        if words[i] == '#' and (words[i+1] >= '1' and words[i+1] <= '4'):
            if words[i+1] == '1':
                pass
            else:
                if words[i+1] == '2':
                    pinyin.insert(insert_pos, 'sp2')
                if words[i+1] == '3':
                    pinyin.insert(insert_pos, 'sp2')
                elif words[i+1] == '4':
                    pinyin.append('sil')
                    break
                insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 2
        elif ord(words[i]) == CODE_ERX:
            if pinyin[insert_pos-1].find('er') != 0: # erhua
                i += 1
            else:
                insert_pos = _update_insert_pos(insert_pos, pinyin)
                i += 1
        # skip non-mandarin characters, including A-Z, a-z, Greece letters, etc.
        elif ord(words[i]) < 0x4E00 or ord(words[i]) > 0x9FA5:
            i += 1
        else:
            insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 1
    return pinyin

def _pinyin_2_initialfinal(py):
    """
    used to split pinyin into intial and final phonemes
    """
    if py[0] == 'a' or py[0] == 'e' or py[0] == 'E' or py[0] == 'o' or py[:2] == 'ng' or \
            py[:2] == 'hm':
        py_initial = ''
        py_final = py
    elif py[0] == 'y':
        py_initial = ''
        if py[1] == 'u' or py[1] == 'v':
            py_final = list(py[1:])
            py_final[0] = 'v'
            py_final = ''.join(py_final)
        elif py[1] == 'i':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'i'
            py_final = ''.join(py_final)
    elif py[0] == 'w':
        py_initial = ''
        if py[1] == 'u':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'u'
            py_final = ''.join(py_final)
    else:
        init_cand = ''
        for init in MANDARIN_INITIAL_LIST:
            init_len = len(init)
            init_cand = py[:init_len]
            if init_cand == init:
                break
        if init_cand == '':
            raise Exception('unexpected')
        py_initial = init_cand
        py_final = py[init_len:]
        if (py_initial in set(['j', 'q', 'x']) and py_final[0] == 'u'):
            py_final = list(py_final)
            py_final[0] = 'v'
            py_final = ''.join(py_final)
    if py_final[-1] == '6':
        py_final = py_final.replace('6', '2')
    return (py_initial, py_final)

def is_all_eng(words):
    #if include mandarin
    for word in words:
        if ord(word) >= 0x4E00 and ord(word) <= 0x9FA5:
            return False
    return True

def pinyin_2_phoneme(pinyin_line, words):
    #chn or chn+eng
    sent_phoneme = ['sp1']
    if not is_all_eng(words):
        sent_py = _pinyin_preprocess(pinyin_line, words)
        for py in sent_py:
            if py[0].isupper() or py in CHN_PHONE_PUNC_LIST:
                sent_phoneme.append(py)
            else:
                initial, final = _pinyin_2_initialfinal(py)
                if initial == '':
                    sent_phoneme.append(final)
                else:
                    sent_phoneme.append(initial)
                    sent_phoneme.append(final)
    else:
        wordlist = words.split(' ')
        word_phonelist = pinyin_line.strip().split('/')
        assert(len(word_phonelist) == len(wordlist))
        i = 0
        while i < len(word_phonelist):
            phone = re.split(r'[ .]', word_phonelist[i])
            for p in phone:
                if p:
                    sent_phoneme.append(p)
            if '/' in wordlist[i]:
                sent_phoneme.append('sp2')
            elif '%' in wordlist[i]:
                if i != len(word_phonelist)-1:
                    sent_phoneme.append('sp2')
                else:
                    sent_phoneme.append('sil')
            i += 1
    return ' '.join(sent_phoneme)



def trans_prosody(file_trans,dic_phoneme):
    
    is_sentid_line = True
    with open(file_trans, encoding='utf-8') as f,\
            open('biaobei_prosody.csv', 'w') as fw:
        for line in f:
            if is_sentid_line:
                sent_id = line.split()[0]
                words = line.split('\t')[1].strip()
            else:
               
                sent_phonemes = pinyin_2_phoneme(line, words)
                
                sent_sent_phonemes_index = ''
                for phonemes in sent_phonemes.split():
                    sent_sent_phonemes_index = sent_sent_phonemes_index+ str(dic_phoneme[phonemes]) + ' '
                    
                sent_sent_phonemes_index = sent_sent_phonemes_index + str(dic_phoneme['~'])# 添加eos
                print(sent_sent_phonemes_index)
                fw.writelines('|'.join([sent_id, sent_phonemes, sent_sent_phonemes_index]) + '\n')
            is_sentid_line = not is_sentid_line

def index_unknown():
    return 0
if __name__ == "__main__":
    
    para = hparams()
    
    wavs = glob.glob(para.path_wav+'/*wav') 
    processing_wavs(wavs,para)
    file_trans = para.file_trans
    
    # 字典文件
    vocab_file = os.path.join(para.path_scp,'vocab')
   
    trans_prosody(file_trans,para.dic_phoneme)
    
    
    
    
    
    
