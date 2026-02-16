from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
import argparse
from torch import no_grad, LongTensor
import logging
import os

logging.getLogger('numba').setLevel(logging.WARNING)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message, speaker_id=None):
    if speaker_id is not None:
        try:
            speaker_id = int(speaker_id)
        except:
            print(str(speaker_id) + ' is not a valid ID!')
            sys.exit(1)
        return speaker_id
    else:
        speaker_id = input(message)
        try:
            speaker_id = int(speaker_id)
        except:
            print(str(speaker_id) + ' is not a valid ID!')
            sys.exit(1)
        return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


def voice_conversion(net_g_ms, hps_ms, speakers, escape, 
                     audio_path=None, original_id=None, target_id=None, out_path=None):
    if audio_path is None:
        audio_path = input('Path of an audio file to convert:\n')
    
    print_speakers(speakers, escape)
    audio = utils.load_audio_to_torch(
        audio_path, hps_ms.data.sampling_rate)

    originnal_id = get_speaker_id('Original speaker ID: ', original_id)
    target_id = get_speaker_id('Target speaker ID: ', target_id)
    
    if out_path is None:
        out_path = input('Path to save: ')

    y = audio.unsqueeze(0)

    spec = spectrogram_torch(y, hps_ms.data.filter_length,
                             hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                             center=False)
    spec_lengths = LongTensor([spec.size(-1)])
    sid_src = LongTensor([originnal_id])

    with no_grad():
        sid_tgt = LongTensor([target_id])
        audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[
            0][0, 0].data.cpu().float().numpy()
    return audio, out_path


def tts_inference(net_g_ms, hps_ms, speakers, escape, 
                  text=None, speaker_id=None, out_path=None,
                  length_scale=1.0, noise_scale=0.667, noise_scale_w=0.8, cleaned=False):
    
    if text is None:
        text = input('Text to read: ')
        if text == '[ADVANCED]':
            text = input('Raw text:')
            print('Cleaned text is:')
            ex_print(_clean_text(text, hps_ms.data.text_cleaners), escape)
            return None, None
    
    # Process text for control tags
    length_scale, text = get_label_value(text, 'LENGTH', length_scale, 'length scale')
    noise_scale, text = get_label_value(text, 'NOISE', noise_scale, 'noise scale')
    noise_scale_w, text = get_label_value(text, 'NOISEW', noise_scale_w, 'deviation of noise')
    cleaned_tag, text = get_label(text, 'CLEANED')
    cleaned = cleaned or cleaned_tag

    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

    print_speakers(speakers, escape)
    speaker_id = get_speaker_id('Speaker ID: ', speaker_id)
    
    if out_path is None:
        out_path = input('Path to save: ')

    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    
    return audio, out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VITS TTS/VC Inference')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path of a VITS model')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path of a config file')
    parser.add_argument('--escape', action='store_true',
                       help='Escape output text')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['tts', 'vc', 'interactive'],
                       default='interactive', help='Run mode: tts, vc, or interactive')
    
    # TTS parameters
    parser.add_argument('--text', type=str, help='Text to synthesize (for TTS mode)')
    parser.add_argument('--speaker_id', type=int, help='Speaker ID for TTS')
    parser.add_argument('--length_scale', type=float, default=1.0, help='Length scale for TTS')
    parser.add_argument('--noise_scale', type=float, default=0.667, help='Noise scale for TTS')
    parser.add_argument('--noise_scale_w', type=float, default=0.8, help='Deviation of noise for TTS')
    parser.add_argument('--cleaned', action='store_true', help='Use cleaned text')
    
    # VC parameters
    parser.add_argument('--input_audio', type=str, help='Input audio path for VC')
    parser.add_argument('--original_id', type=int, help='Original speaker ID for VC')
    parser.add_argument('--target_id', type=int, help='Target speaker ID for VC')
    
    # Output
    parser.add_argument('--output', '-o', type=str, help='Output audio path')
    
    # Emotion parameters (for emotion_embedding models)
    parser.add_argument('--emotion_model', type=str, help='Path to w2v2 emotion model (for emotion_embedding)')
    parser.add_argument('--emotion_ref', type=str, help='Path to emotion reference audio or npy file')
    
    args = parser.parse_args()
    
    escape = args.escape
    model = args.model
    config = args.config

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    # Handle different modes
    if n_symbols != 0:
        if not emotion_embedding:
            if args.mode == 'tts':
                # TTS mode - non-interactive
                if args.text is None or args.speaker_id is None or args.output is None:
                    print("Error: TTS mode requires --text, --speaker_id, and --output")
                    sys.exit(1)
                
                audio, out_path = tts_inference(
                    net_g_ms, hps_ms, speakers, escape,
                    text=args.text,
                    speaker_id=args.speaker_id,
                    out_path=args.output,
                    length_scale=args.length_scale,
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    cleaned=args.cleaned
                )
                
                if audio is not None:
                    write(out_path, hps_ms.data.sampling_rate, audio)
                    print(f'Successfully saved to {out_path}')
                
            elif args.mode == 'vc':
                # VC mode - non-interactive
                if args.input_audio is None or args.original_id is None or args.target_id is None or args.output is None:
                    print("Error: VC mode requires --input_audio, --original_id, --target_id, and --output")
                    sys.exit(1)
                
                audio, out_path = voice_conversion(
                    net_g_ms, hps_ms, speakers, escape,
                    audio_path=args.input_audio,
                    original_id=args.original_id,
                    target_id=args.target_id,
                    out_path=args.output
                )
                
                write(out_path, hps_ms.data.sampling_rate, audio)
                print(f'Successfully saved to {out_path}')
                
            else:  # interactive mode
                while True:
                    choice = input('TTS or VC? (t/v):')
                    if choice == 't':
                        text = input('Text to read: ')
                        if text == '[ADVANCED]':
                            text = input('Raw text:')
                            print('Cleaned text is:')
                            ex_print(_clean_text(
                                text, hps_ms.data.text_cleaners), escape)
                            continue

                        length_scale, text = get_label_value(
                            text, 'LENGTH', 1, 'length scale')
                        noise_scale, text = get_label_value(
                            text, 'NOISE', 0.667, 'noise scale')
                        noise_scale_w, text = get_label_value(
                            text, 'NOISEW', 0.8, 'deviation of noise')
                        cleaned, text = get_label(text, 'CLEANED')

                        stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                        print_speakers(speakers, escape)
                        speaker_id = get_speaker_id('Speaker ID: ')
                        out_path = input('Path to save: ')

                        with no_grad():
                            x_tst = stn_tst.unsqueeze(0)
                            x_tst_lengths = LongTensor([stn_tst.size(0)])
                            sid = LongTensor([speaker_id])
                            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                                   noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

                    elif choice == 'v':
                        audio, out_path = voice_conversion(net_g_ms, hps_ms, speakers, escape)

                    write(out_path, hps_ms.data.sampling_rate, audio)
                    print('Successfully saved!')
                    ask_if_continue()
                    
        else:  # emotion_embedding enabled
            import librosa
            import numpy as np
            from torch import FloatTensor
            import audonnx
            
            if args.mode != 'interactive' and args.emotion_model is None:
                print("Error: Emotion model required for non-interactive mode with emotion_embedding")
                sys.exit(1)
            
            w2v2_folder = args.emotion_model if args.emotion_model else input('Path of a w2v2 dimensional emotion model: ')
            w2v2_model = audonnx.load(os.path.dirname(w2v2_folder))
            
            if args.mode == 'tts':
                # TTS mode with emotion
                if args.text is None or args.speaker_id is None or args.emotion_ref is None or args.output is None:
                    print("Error: TTS mode requires --text, --speaker_id, --emotion_ref, and --output")
                    sys.exit(1)
                
                # Process emotion reference
                if args.emotion_ref.endswith('.npy'):
                    emotion = np.load(args.emotion_ref)
                    emotion = FloatTensor(emotion).unsqueeze(0)
                else:
                    audio16000, sampling_rate = librosa.load(
                        args.emotion_ref, sr=16000, mono=True)
                    emotion = w2v2_model(audio16000, sampling_rate)['hidden_states']
                    emotion = FloatTensor(emotion)
                
                # Process text
                length_scale, text = get_label_value(args.text, 'LENGTH', args.length_scale, 'length scale')
                noise_scale, text = get_label_value(args.text, 'NOISE', args.noise_scale, 'noise scale')
                noise_scale_w, text = get_label_value(args.text, 'NOISEW', args.noise_scale_w, 'deviation of noise')
                cleaned, text = get_label(text, 'CLEANED')
                cleaned = cleaned or args.cleaned
                
                stn_tst = get_text(text, hps_ms, cleaned=cleaned)
                
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([args.speaker_id])
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                          noise_scale_w=noise_scale_w, length_scale=length_scale,
                                          emotion_embedding=emotion)[0][0, 0].data.cpu().float().numpy()
                
                write(args.output, hps_ms.data.sampling_rate, audio)
                print(f'Successfully saved to {args.output}')
                
            elif args.mode == 'vc':
                audio, out_path = voice_conversion(net_g_ms, hps_ms, speakers, escape,
                                                   audio_path=args.input_audio,
                                                   original_id=args.original_id,
                                                   target_id=args.target_id,
                                                   out_path=args.output)
                write(out_path, hps_ms.data.sampling_rate, audio)
                print(f'Successfully saved to {out_path}')
                
            else:  # interactive mode with emotion
                while True:
                    choice = input('TTS or VC? (t/v):')
                    if choice == 't':
                        text = input('Text to read: ')
                        if text == '[ADVANCED]':
                            text = input('Raw text:')
                            print('Cleaned text is:')
                            ex_print(_clean_text(
                                text, hps_ms.data.text_cleaners), escape)
                            continue

                        length_scale, text = get_label_value(
                            text, 'LENGTH', 1, 'length scale')
                        noise_scale, text = get_label_value(
                            text, 'NOISE', 0.667, 'noise scale')
                        noise_scale_w, text = get_label_value(
                            text, 'NOISEW', 0.8, 'deviation of noise')
                        cleaned, text = get_label(text, 'CLEANED')

                        stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                        print_speakers(speakers, escape)
                        speaker_id = get_speaker_id('Speaker ID: ')

                        emotion_reference = input('Path of an emotion reference: ')
                        if emotion_reference.endswith('.npy'):
                            emotion = np.load(emotion_reference)
                            emotion = FloatTensor(emotion).unsqueeze(0)
                        else:
                            audio16000, sampling_rate = librosa.load(
                                emotion_reference, sr=16000, mono=True)
                            emotion = w2v2_model(audio16000, sampling_rate)[
                                'hidden_states']
                            emotion_reference = re.sub(
                                r'\..*$', '', emotion_reference)
                            np.save(emotion_reference, emotion.squeeze(0))
                            emotion = FloatTensor(emotion)

                        out_path = input('Path to save: ')

                        with no_grad():
                            x_tst = stn_tst.unsqueeze(0)
                            x_tst_lengths = LongTensor([stn_tst.size(0)])
                            sid = LongTensor([speaker_id])
                            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                                   length_scale=length_scale, emotion_embedding=emotion)[0][0, 0].data.cpu().float().numpy()

                    elif choice == 'v':
                        audio, out_path = voice_conversion(net_g_ms, hps_ms, speakers, escape)

                    write(out_path, hps_ms.data.sampling_rate, audio)
                    print('Successfully saved!')
                    ask_if_continue()
                    
    else:  # hubert-soft model
        if args.mode == 'vc' or args.mode == 'interactive':
            hubert_model_path = args.model if args.mode == 'vc' else input('Path of a hubert-soft model: ')
            from hubert_model import hubert_soft
            hubert = hubert_soft(hubert_model_path)
            
            if args.mode == 'vc':
                # Hubert VC mode
                if args.input_audio is None or args.target_id is None or args.output is None:
                    print("Error: VC mode requires --input_audio, --target_id, and --output")
                    sys.exit(1)
                
                import librosa
                import numpy as np
                from torch import inference_mode, FloatTensor
                
                if use_f0:
                    audio, sampling_rate = librosa.load(
                        args.input_audio, sr=hps_ms.data.sampling_rate, mono=True)
                    audio16000 = librosa.resample(
                        audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio16000, sampling_rate = librosa.load(
                        args.input_audio, sr=16000, mono=True)
                
                length_scale = args.length_scale
                noise_scale = args.noise_scale
                noise_scale_w = args.noise_scale_w
                
                with inference_mode():
                    units = hubert.units(FloatTensor(audio16000).unsqueeze(
                        0).unsqueeze(0)).squeeze(0).numpy()
                    if use_f0:
                        f0_scale = args.length_scale  # Reusing length_scale for f0_scale
                        f0 = librosa.pyin(audio, sr=sampling_rate,
                                          fmin=librosa.note_to_hz('C0'),
                                          fmax=librosa.note_to_hz('C7'),
                                          frame_length=1780)[0]
                        target_length = len(units[:, 0])
                        f0 = np.nan_to_num(np.interp(np.arange(0, len(f0)*target_length, len(f0))/target_length,
                                                     np.arange(0, len(f0)), f0)) * f0_scale
                        units[:, 0] = f0 / 10
                
                stn_tst = FloatTensor(units)
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([args.target_id])
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.float().numpy()
                
                write(args.output, hps_ms.data.sampling_rate, audio)
                print(f'Successfully saved to {args.output}')
                
            else:  # interactive mode
                while True:
                    audio_path = input('Path of an audio file to convert:\n')

                    if audio_path != '[VC]':
                        import librosa
                        if use_f0:
                            audio, sampling_rate = librosa.load(
                                audio_path, sr=hps_ms.data.sampling_rate, mono=True)
                            audio16000 = librosa.resample(
                                audio, orig_sr=sampling_rate, target_sr=16000)
                        else:
                            audio16000, sampling_rate = librosa.load(
                                audio_path, sr=16000, mono=True)

                        print_speakers(speakers, escape)
                        target_id = get_speaker_id('Target speaker ID: ')
                        out_path = input('Path to save: ')
                        length_scale, out_path = get_label_value(
                            out_path, 'LENGTH', 1, 'length scale')
                        noise_scale, out_path = get_label_value(
                            out_path, 'NOISE', 0.1, 'noise scale')
                        noise_scale_w, out_path = get_label_value(
                            out_path, 'NOISEW', 0.1, 'deviation of noise')

                        from torch import inference_mode, FloatTensor
                        import numpy as np
                        with inference_mode():
                            units = hubert.units(FloatTensor(audio16000).unsqueeze(
                                0).unsqueeze(0)).squeeze(0).numpy()
                            if use_f0:
                                f0_scale, out_path = get_label_value(
                                    out_path, 'F0', 1, 'f0 scale')
                                f0 = librosa.pyin(audio, sr=sampling_rate,
                                                  fmin=librosa.note_to_hz('C0'),
                                                  fmax=librosa.note_to_hz('C7'),
                                                  frame_length=1780)[0]
                                target_length = len(units[:, 0])
                                f0 = np.nan_to_num(np.interp(np.arange(0, len(f0)*target_length, len(f0))/target_length,
                                                             np.arange(0, len(f0)), f0)) * f0_scale
                                units[:, 0] = f0 / 10

                        stn_tst = FloatTensor(units)
                        with no_grad():
                            x_tst = stn_tst.unsqueeze(0)
                            x_tst_lengths = LongTensor([stn_tst.size(0)])
                            sid = LongTensor([target_id])
                            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                                   noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.float().numpy()

                    else:
                        audio, out_path = voice_conversion(net_g_ms, hps_ms, speakers, escape)

                    write(out_path, hps_ms.data.sampling_rate, audio)
                    print('Successfully saved!')
                    ask_if_continue()
