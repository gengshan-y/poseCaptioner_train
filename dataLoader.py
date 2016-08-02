import csv
import numpy as np
import os
import sys
import random
random.seed(3)
import h5py
from hdf5_npstreamsequence_generator import SequenceGenerator

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<en_unk>'
FEAT_DIM= 32  # 32 dim pose features
BUFFER_SIZE = 32 # number of streams for a batch of data

class fc7FrameSequenceGenerator(SequenceGenerator):
    
    def dump_video_file(self, vidid_order_file, frame_seq_label_file):
        print 'Dumping vidid order to file: %s' % vidid_order_file
        with open(vidid_order_file,'wb') as vidid_file:
            vidid_file.write('vidid\tword_count\tframe_count\ttotal_count\n')
            for vidid, line in self.lines:
                word_count = len(line.split())
                frame_count = len(self.vid_framefeats[vidid])
                total_count = word_count +frame_count
                vidid_file.write('%s\t%d\t%d\t%d\n' % (vidid, word_count, frame_count, total_count))
        print 'Done.'
    
    def streams_exhausted(self):
        return self.num_resets > 0

    
    def next_line(self):
        num_lines = float(len(self.lines))
        if self.line_index == 1 or self.line_index == num_lines or self.line_index % 100 == 0:
            print 'Processed %d/%d (%f%%) lines' % (self.line_index, num_lines,
                                              100 * self.line_index / num_lines)
        self.line_index += 1
        if self.line_index == num_lines:
            self.line_index = 0
            self.num_resets += 1
    
    def line_to_stream(self, sentence):
        stream = []
        for word in sentence.split():
            word = word.strip()
            if word in self.vocabulary:
                stream.append(self.vocabulary[word])
            else:  # unknown word; append UNK
                stream.append(self.vocabulary[UNK_IDENTIFIER])
        # increment the stream -- 0 will be the EOS character
        stream = [s + 1 for s in stream]
        return stream

    
    # we have pooled fc7 features already in the file
    def get_streams(self):
        vidid, line = self.lines[self.line_index]
        assert vidid in self.vid_framefeats
        feats_vgg_fc7 = self.vid_framefeats[vidid] # list of fc7 feats for all frames
        num_frames = len(feats_vgg_fc7)
        stream = self.line_to_stream(line)  # [27, 1026, 70, 17, 1, 1, 9, 64, 4, 23, 6, 1194]
        num_words = len(stream)
        pad = self.max_words - (num_words + 1 + num_frames) if self.pad else 0
        
        truncated = False
        # print '{0} #frames: {1} #words: {2}'.format(vidid, num_frames, num_words)
        if pad < 0:
            # print '{0} #frames: {1} #words: {2}'.format(vidid, num_frames, num_words)
            # truncate frames
            if (num_words + 1) > self.max_words:  
                stream = stream[:20]  # truncate words to 20 if there are too many words
            num_frames = self.max_words - (len(stream)+1)
            truncated = True
            pad = 0
            # print 'Truncated_{0}: #frames: {1} #words: {2}'.format(vidid, num_frames, len(stream))
            self.num_truncates += truncated
        
        # reverse the string
        if self.reverse:
            stream.reverse()
        
        if pad > 0: self.num_pads += 1
        
        self.num_outs += 1   # {frames}{x}{words}
        out = {}
        out['cont_sentence'] = [0] + [1] * (num_frames +len(stream)) + [0] * pad  # all sequence indicator
        out['input_sentence'] = [0] * num_frames + [0] + stream + [0] * pad  # input caption
        out['target_sentence'] = [-1] * num_frames + stream + [0] + [-1] * pad  # output caption
        # For encoder-decoder model
        out['cont_img'] = [0] + [1] * (num_frames - 1) + [0] * (len(stream) + 1 + pad)  # image exist indicator
        out['cont_sen'] = [0] * (num_frames + 1) + [1] * len(stream) + [0] * pad  # sent exist indicator
        out['encoder_to_decoder'] = [0] * (num_frames - 1) + [1] + [0] * (len(stream) + 1 + pad)  # end of frames indicator
        out['stage_indicator'] = [0] * num_frames + [1] * (len(stream) + 1 + pad)  # indicate ouput stage
        out['inv_stage_indicator'] = [1] * num_frames + [0] * (len(stream) + 1 + pad)  # indicate input stage
        # fc7 features
        out['frame_fc7'] = []  # input frame features
        
        for frame_feat in feats_vgg_fc7[:num_frames]:
            feat_fc7 = map(float, frame_feat.split(','))
            out['frame_fc7'].append(np.array(feat_fc7).reshape(1, len(feat_fc7)))
        # pad last frame for the length of the sentence
        num_img_pads = len(out['input_sentence']) - num_frames
        zero_padding = np.zeros(len(feat_fc7)).reshape(1, len(feat_fc7))

        for padframe in range(num_img_pads):
            out['frame_fc7'].append(zero_padding)
            
        assert len(out['frame_fc7'])==len(out['input_sentence'])
        self.next_line()

        return out
            
        
    def init_vocab_from_file(self, vocab_filedes):
        # initialize the vocabulary with the UNK word
        self.vocabulary = {UNK_IDENTIFIER: 0}
        self.vocabulary_inverted = [UNK_IDENTIFIER]
        num_words_dataset = 0
        for line in vocab_filedes.readlines():
            split_line = line.split()
            word = split_line[0]
            # print word
            if unicode(word) == UNK_IDENTIFIER:
                continue
            else:
                assert word not in self.vocabulary
            num_words_dataset += 1
            self.vocabulary[word] = len(self.vocabulary_inverted)
            self.vocabulary_inverted.append(word)
        num_words_vocab = len(self.vocabulary.keys())
        print ('Initialized vocabulary from file with %d unique words ' +
           '(from %d total words in dataset).') % \
          (num_words_vocab, num_words_dataset)
        assert len(self.vocabulary_inverted) == num_words_vocab

    
    def init_vocabulary(self, vocab_filename):
        print "Initializing the vocabulary."
        if os.path.isfile(vocab_filename):
            with open(vocab_filename, 'rb') as vocab_file:
                self.init_vocab_from_file(vocab_file)
        else:
            print('error')
            # self.init_vocabulary_from_data(vocab_filename)

    def __init__(self, filenames, batch_num_streams=1, vocab_filename=None,
               max_words=80, align=True, shuffle=True, pad=True,
               truncate=True, reverse=False):  # reverse - reverse the words, 
                                               # shuffle lines if true
                                               # must truncate
                                               # pad word+frame with 0 if pad
                                               # align means to pad the last buffer
        self.max_words = max_words
        self.reverse = reverse
        self.array_type_inputs = {}  # stream inputs that are arrays
        
        self.array_type_inputs['frame_fc7'] = FEAT_DIM 

        self.lines = []
        num_empty_lines = 0
        self.vid_framefeats = {}  # listofdict [{}]
        framefeatfile = filenames[0]
        sentfile = filenames[1]
        
        """ Read pose data """
        print 'Reading frame features from file: %s' % framefeatfile
        with open(framefeatfile, 'rb') as featfd:
            # each line has the fc7 for 1 frame in video
            pool_csv = csv.reader(featfd)
            pool_csv = list(pool_csv)

            for line in pool_csv:
                id_framenum = line[0]
                video_id = id_framenum.rsplit('_', 1)[0]
                frameData = np.array(line[1:])

        
                ''' Get hdf5 data '''
                if video_id not in self.vid_framefeats:
                    self.vid_framefeats[video_id]=[]
                # self.vid_framefeats[video_id].append(','.join(line[1:]))
                self.vid_framefeats[video_id].append(','.join(str(e) for e in frameData))  # {clipname: [pose1, pose2, pose3, ...]}

        ## reset max_words based on maximum frames in the video 
        """ Read caption data """
        print 'Reading sentences in: %s' % sentfile
        with open(sentfile, 'r') as sentfd:
            for line in sentfd:
                line = line.strip()
                id_sent = line.split('\t')
                if len(id_sent)<2:
                    num_empty_lines += 1
                    continue
                self.lines.append((id_sent[0], id_sent[1]))  # (clipName, caption)
            
        if num_empty_lines > 0:
            print 'Warning: ignoring %d empty lines.' % num_empty_lines
        

        self.line_index = 0
        self.num_resets = 0
        self.num_truncates = 0
        self.num_pads = 0
        self.num_outs = 0
        self.frame_list = []
        self.vocabulary = {}
        self.vocabulary_inverted = []
        self.vocab_counts = []
        
        """ initialize vocabulary """
        
        self.init_vocabulary(vocab_filename)
        SequenceGenerator.__init__(self)
        self.batch_num_streams = batch_num_streams  # needed in hdf5 to seq
        # make the number of image/sentence pairs a multiple of the buffer size
        # so each timestep of each batch is useful and we can align the images
        if align:
            num_pairs = len(self.lines)
            remainder = num_pairs % BUFFER_SIZE
            if remainder > 0:
                num_needed = BUFFER_SIZE - remainder
                for i in range(num_needed):
                    choice = random.randint(0, num_pairs - 1)
                    self.lines.append(self.lines[choice])
            assert len(self.lines) % BUFFER_SIZE == 0
        if shuffle:
            random.shuffle(self.lines)
        self.pad = pad
        self.truncate = truncate
        self.negative_one_padded_streams = frozenset(('target_sentence'))  # what's that?
