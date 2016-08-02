from utils import gen_stats, vocab_inds_to_sentence
import json
title2num = json.load(open('/home/gengshan/public_html/data/ori_data_list.json'\
                           , 'r'))

def to_html_row(columns, header=False):
    out= '<tr>'

    for column in columns:
        if header: out += '<th>'
        else: out += '<td>'
        try:
            if int(column) < 1e8 and int(column) == float(column):
                out += '%d' % column
            else:
                out += '%0.04f' % column
        except:
            out += '%s' % column
        if header: out += '</th>'
        else: out += '</td>'

    out += '</tr>'
    return out

def to_html_output(outputs, vocab):
    out = ''

    for video_id, captions in outputs.iteritems():
        for c in captions:
            if not 'stats' in c:
                c['stats'] = gen_stats(c['prob'])
            
        ''' Sort captions by log probability '''
        if 'normed_perplex' in captions[0]['stats']:
            captions.sort(key=lambda c: c['stats']['normed_perplex'])
        else:
            captions.sort(key=lambda c: -c['stats']['log_p_word'])
            
        out += '<video width="640" height="480" controls=""' + \
           'src="data/clip/' + \
            video_id.rsplit('_', 1)[0] + '/' + \
            video_id + '.mp4"></video>\n'
        out += '<table border="1">\n'
        # column_names = ('Source', '#Words', 'Perplexity/Word', 'Caption')
        column_names = ('Clip_id', 'Caption', 'True', 'Log_p_word')
        out += '%s\n' % to_html_row(column_names, header=True)
        
        for c in captions:
            caption, gt, source, stats = \
              c['caption'], c['gt'], c['source'], c['stats']
            caption_string = vocab_inds_to_sentence(vocab, caption)
            if gt:
                source = 'ground truth'
                if 'correct' in c:
                    caption_string = '<font color="%s">%s</font>' % \
                      ('green' if c['correct'] else 'red', caption_string)
                else:
                    caption_string = '<em>%s</em>' % caption_string
            else:
                if source['type'] == 'beam':
                    source = 'beam (size %d)' % source['beam_size']
                elif source['type'] == 'sample':
                    source = 'sample (temp %f)' % source['temp']
                else:
                    raise Exception('Unknown type: %s' % source['type'])
                caption_string = '<strong>%s</strong>' % caption_string
            
            
            with open('/data2/gengshan/clip/' + \
               video_id.rsplit('_', 1)[0] + '/' + video_id + '.txt', 'r') as tmpSubFile:
                tmpSub = ''.join(tmpSubFile.readlines())
            tmpClipNum = str(title2num[video_id.rsplit('_', 1)[0]]) + '_' + video_id.rsplit('_', 1)[1]
            # columns = (source, stats['length'] - 1, stats['perplex_word'], caption_string, tmpSub)
            columns = (tmpClipNum, caption_string, tmpSub, stats['log_p_word'])
            out += '%s\n' % to_html_row(columns)
            
        out += '</table>\n'
        out += '<br>\n\n' 
        out += '<br>' * 2
    
    out.replace('<unk>', 'UNK')  # sanitize...
    return out
