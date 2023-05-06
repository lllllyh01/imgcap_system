from mmap import ACCESS_WRITE
from torch import long
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torch.utils.data
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from datasets import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval
from sklearn.metrics import f1_score
import numpy as np
import random

# Parameters
# data_folder = './peir_224_12_28'  # folder with data files saved by create_input_files.py
# data_name = 'coco_1_cap_per_img_2_min_word_freq'  # base name shared by data files
# # checkpoint = 'BEST_checkpoint_coco_1_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
# # checkpoint='./BEST_2checkpoint_coco_1_cap_per_img_5_min_word_freq.pth.tar'
# checkpoint='./BEST_3checkpoint_coco_1_cap_per_img_2_min_word_freq.pth.tar'
# word_map_file = './peir_224_12_28/WORDMAP_coco_1_cap_per_img_2_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
# tag_map_file='./peir_224_12_28/TAGMAP_coco_1_cap_per_img_2_min_word_freq.json'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

data_folder = './peir_224_12_28'  # folder with data files saved by create_input_files.py
# data_folder = './iu_output_file_freq_5_224'  # folder with data files saved by create_input_files.py
data_name = 'coco_1_cap_per_img_2_min_word_freq'  # base name shared by data files
# checkpoint = 'BEST_checkpoint_coco_1_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = data_folder + '/WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
tag_map_file = data_folder + '/TAGMAP_' + data_name + '.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
lbda=0.5


def tag_origin(tag):
    tag_ori = []
    for i in tag:
        i = i.item()
        # print(i)
        # print(type(i))
        if i != 0 and i != 211 and i != 212:
            tag_ori.append(rev_tag_map[i])
    
    return ",".join(tag_ori) #string


def tags_yn_convert(mlp_out):
    threshold = 0.3
    pred_tags_yn = torch.zeros_like(mlp_out).long()

    for i, beam in enumerate(mlp_out):
        for j, val in enumerate(beam):
            if val > threshold:
                pred_tags_yn[i][j] = 1
    
    return pred_tags_yn


def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]


def evaluate(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    tag_ori_list = []
    pred_tags_yn_list = np.zeros(shape=(357,209))
    tags_yn_list = np.zeros(shape=(357,209))
    # for i, (image, caps, caplens, allcaps,tags,taglens,tags_yn) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
    for i, (image, caps, caplens, allcaps,tags,taglens,tags_yn) in enumerate(loader):
        # print("tags shape:", tags.shape) # (1, 52)
        # print("tags:", tags) # tag of current evaluating image
        # print("tags_yn:", tags_yn) # tag(one-hot) of current evaluating image
        # print("tags_yn shape:", tags_yn.shape) # (1, 209)
        
        tag_ori_list.append(tag_origin(tags[0]))
        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        tags=tags.to(device)
        # if(i==0):
        #     print(tags)
        taglens=taglens.to(device)
        tags_yn=tags_yn.to(device)
        # for j in range(len(tags_yn[0])):
        #     tags_yn_list[i][j] = tags_yn[0][j]

        # Encode
        # encoder_out,h_c= encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out,hc=encoder(image)
        # enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(-1)

        # # Flatten encoding
        # encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k,encoder_dim)  # (k, encoder_dim)

        ## mlp_out=decoder.mlp(encoder_out) # (k, tag_size)
        ## pred_tags_yn = tags_yn_convert(mlp_out[:1]) # (1, tag_size)
        # print("pred_tags_yn shape:", pred_tags_yn.shape)
        ## for j in range(len(pred_tags_yn[0])):
        ##     pred_tags_yn_list[i][j] = pred_tags_yn[0][j]

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1,c1=decoder.init_hidden_state(k)
        h2,c2=decoder.init_hidden_state(k)
        decoder_dim=h2.size(1)
        #h3,c3=decoder.init_hidden_state(k)

        ## encoded_tag=decoder.tag_convert(mlp_out,rev_tag_map,tag_map)
        # print("mlp_out shape:", mlp_out.shape) # (k, tag_size(209))
        # print("encoded_tag shape:", encoded_tag.shape) # (k, tag_len(10))
        ## tagembedding=decoder.tagembedding(encoded_tag) # (k, tag_len(10), tagembed(32))
        # print("tagembedding shape:", tagembedding.shape)
        # tagembedding_decoder=decoder.tagembedding(pred_tags_yn)[0]

        # _,h_c_tag=decoder.global_tag(tagembedding)
        # h1,c1=h_c_tag
        # h1=h1.squeeze(0)
        # c1=c1.squeeze(0)
        # decoder_dim=h1.size(1)


        # _,h_c_tag=decoder.global_tag(tagembedding)
        # h,c=h_c_tag
        # h=h.squeeze(0)
        # c=c.squeeze(0)

        # h_t,c_t=h_c
        # h_t=h_t.squeeze(0)
        # c_t=c_t.squeeze(0)

        # h1=h_t
        # c1=c_t
        # h1=h_t.expand(k,decoder_dim)
        # c1=c_t.expand(k,decoder_dim)

        # h=h+h_t
        # c=c+c_t

        # h2=hc
        # c2=hc
        # h2=h2.expand(k,decoder_dim)
        # c2=c2.expand(k,decoder_dim)

        # h=h+h2
        # c=c+c2

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            # ctx= decoder.attention(encoder_out,h1,h)
            # h1,c1=decoder.decode_step(torch.cat([embeddings,ctx,h1],dim=1),(h1,c1))
            # ctx1=decoder.attention(encoder_out,h1,h)
            # h2,c2=decoder.sent_lstm(torch.cat([ctx1,h1],dim=1),(h2,c2))
            # scores1=decoder.fc(h2)
            # if step==1:
            #     scores=scores1
            # else:
            #     scores=scores1+lbda*scores2
            # ctx2=decoder.attention(encoder_out,h2,h)
            # h3,c3=decoder.sent_lstm(torch.cat([ctx2,h2],dim=1),(h2,c2))
            # scores2=decoder.fc(h3)
            ctx1, _= decoder.attention(encoder_out,h1)
            h1,c1=decoder.sent_lstm(torch.cat([ctx1,h1],dim=1),(h1,c1))
            h2,c2=decoder.decode_step1(torch.cat([embeddings, ctx1,h1], dim=1),(h2, c2))
            scores1=decoder.fc(h2)
            if step==1:
                scores=scores1
            else:
                scores=scores1+lbda*scores2 
            ctx2, _=decoder.attention(encoder_out,h2)
            #h3,c3=decoder.decode_step(torch.cat([embeddings,ctx2,h2],dim=1),(h3,c3))
            # print("embeddings shape:", embeddings.shape) # (1, embedding_dim)
            # print("tagembedding_decoder shape:", tagembedding_decoder.shape) # (tag_size, embedding_dim)
            # h3,c3=decoder.decode_step2(torch.cat([embeddings,ctx2,h2,tagembedding_decoder[:k,:]],dim=1),(h2,c2))
            h3,c3=decoder.decode_step2(torch.cat([embeddings,ctx2,h2],dim=1),(h2,c2))

            scores2=decoder.fc(h3)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            ## tagembedding = tagembedding[prev_word_inds[incomplete_inds]]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            # h3 = h3[prev_word_inds[incomplete_inds]]
            # c3 = c3[prev_word_inds[incomplete_inds]]
            scores2=scores2[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)


            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
            
        if len(complete_seqs_scores)==0:
            continue
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        # img_caps = allcaps[0].tolist()
        # img_captions = list(
        #     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
        #         img_caps))  # remove <start> and pads
        
        # references.append(img_captions)
        # #print(references)

        # # Hypotheses
        # hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        img_caps = [' '.join(c) for c in img_captions]
        #print(img_caps)
        references.append(img_caps)
        #print(references)

        # Hypotheses
        hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypothesis = ' '.join(hypothesis)
        #print(hypothesis)
        #print(hypothesis)
        hypotheses.append(hypothesis)
        #print(hypotheses)
        assert len(references) == len(hypotheses)


    # m=0
    # n=0
    # r=0
    # t=0
    # hyp=[]
    # ref=[]
    # protion=0
    # # print(len(hypotheses))
    # for i in range(len(hypotheses)):
    #     hyp=hypotheses[i].split('.')
    #     ref=''.join(references[i]).split('.')
    #     for sen in ref:
    #         words=sen.split()
    #         t=t+1
    #         for word in words:
    #             if word in ['no','normal','clear','stable']:
    #                 r=r+1
    #                 break
    #     for sen in hyp:
    #         words=sen.split()
    #         n=n+1
    #         for word in words:
    #             if word in ['no','normal','clear','stable']:
    #                 m=m+1
    #                 break
    # protion=m/n
    # truth=r/t
    # print("nor of truth")
    # print(r)
    # print(t)
    # print(truth)
    # print("nor of prediction")
    # print(m)
    # print(n)
    # print(protion)
    # Calculate BLEU-4 scores
    #bleu4 = corpus_bleu(references, hypotheses)
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    for key in metrics_dict.keys():
        metrics_dict[key] = format(metrics_dict[key], '.5f')
    ## tag_f1 = f1_score(y_pred=pred_tags_yn_list, y_true=tags_yn_list, average="micro")
    ## metrics_dict['Tag_F1(MICRO)'] = tag_f1
    ## tag_precision = Precision(y_pred=pred_tags_yn_list, y_true=tags_yn_list)
    ## metrics_dict['Tag_Precision'] = tag_precision

    # for i in range(5):
    #     print(f"{i}-refer tag: {tag_ori_list[i]}")
    #     print(f"  hypo tag: {tag_origin(tagembedding_decoder[0])}")
    #     print(f"  reference:{references[i]}")
    #     print(f"  hypothesis:{hypotheses[i]}")

    # if beam_size == 5:
    # normal_cnt = 0
    # abnormal_cnt = 0
    # i = 0
    # while normal_cnt <= 5 or abnormal_cnt <= 5:
    #     if 'normal' in tag_ori_list[i] and normal_cnt < 5:
    #         normal_cnt += 1
    #         print(f"{i}-tag: {tag_ori_list[i]}")
    #         print(f"  reference:{references[i]}")
    #         print(f"  hypothesis:{hypotheses[i]}")
    #         print()
    #     elif 'normal' not in tag_ori_list[i] and abnormal_cnt < 5:
    #         abnormal_cnt += 1
    #         print(f"{i}-tag: {tag_ori_list[i]}")
    #         print(f"  reference:{references[i]}")
    #         print(f"  hypothesis:{hypotheses[i]}")
    #         print()

    #     i += 1
    for i in range(20):
        tmp = random.randint(0, 499)
        print(f"{i}-tag: {tag_ori_list[tmp]}")
        print(f"  reference:{references[tmp]}")
        print(f"  hypothesis:{hypotheses[tmp]}")
        print()

    return metrics_dict



if __name__ == '__main__':
    # print('dataset: iu\t'
    #       'model: vit(no mlp & no hc)+mhatt\t'
    #       'parameters: head_num=8, dim=512, dropout=0.5')
    for epoch in range(1):
        # Load model
        print("Epoch =", epoch)
        checkpoint='./checkpoint/' + str(epoch) + '_vit_dw_t_checkpoint_' + data_name + '.pth.tar'
        checkpoint = torch.load(checkpoint,map_location = device)
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()

        nlgeval=NLGEval()

        # Load word map (word2ix)
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        with open(tag_map_file, 'r') as j:
            tag_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}
        rev_tag_map = {v: k for k, v in tag_map.items()}
        vocab_size = len(word_map)

        # Normalization transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        # print('beamsize=1:')
        # beam_size = 1
        # print(evaluate(beam_size))

        print('beamsize=2:')
        beam_size = 2
        print(evaluate(beam_size))
        print()
