import gradio as gr
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import torch
import re
import os
import tempfile
import logging
from torch import no_grad, LongTensor
import numpy as np

logging.getLogger('numba').setLevel(logging.WARNING)

# 全局变量存储模型和配置
model_global = None
hps_global = None
speakers_global = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认模型路径
DEFAULT_MODEL_DIR = "/kaggle/working/ema"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "model.pth")  # 请根据实际文件名修改
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_MODEL_DIR, "config.json")  # 请根据实际文件名修改

def find_model_files(directory):
    """在目录中查找模型和配置文件"""
    model_file = None
    config_file = None
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith('.pth'):
                model_file = os.path.join(directory, file)
            elif file.endswith('.json'):
                config_file = os.path.join(directory, file)
    
    return model_file, config_file

def load_model(model_path, config_path):
    """加载VITS模型和配置"""
    global model_global, hps_global, speakers_global
    
    try:
        # 如果路径为空，尝试使用默认路径
        if not model_path or not config_path:
            default_model, default_config = find_model_files(DEFAULT_MODEL_DIR)
            if default_model and default_config:
                model_path = default_model
                config_path = default_config
                status_msg = f"使用默认模型：{os.path.basename(model_path)}\n"
            else:
                return None, "❌ 未指定模型文件且在默认路径未找到模型"
        else:
            status_msg = ""
        
        # 加载配置
        hps = utils.get_hparams_from_file(config_path)
        
        # 获取说话人列表
        speakers = hps.speakers if 'speakers' in hps.keys() else ['0']
        n_speakers = hps.data.n_speakers if 'n_speakers' in hps.data.keys() else 0
        n_symbols = len(hps.symbols) if 'symbols' in hps.keys() else 0
        
        # 初始化模型
        net_g = SynthesizerTrn(
            n_symbols,
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=n_speakers,
            **hps.model).to(device)
        
        net_g.eval()
        
        # 修复：load_checkpoint只需要两个参数
        utils.load_checkpoint(model_path, net_g)
        
        # 保存到全局变量
        model_global = net_g
        hps_global = hps
        speakers_global = speakers
        
        return speakers, f"{status_msg}✅ 模型加载成功！发现 {len(speakers)} 个说话人"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 模型加载失败：{str(e)}"

def get_speaker_list():
    """获取说话人列表供下拉框使用"""
    if speakers_global:
        return [(f"{name} (ID:{idx})", idx) for idx, name in enumerate(speakers_global)]
    return [("无说话人", 0)]

def process_text(text, length_scale, noise_scale, noise_scale_w):
    """处理文本中的控制标签"""
    if text is None or text == "":
        return None, length_scale, noise_scale, noise_scale_w, False
    
    # 提取控制标签
    length_scale, text = get_label_value(text, 'LENGTH', length_scale, 'length scale')
    noise_scale, text = get_label_value(text, 'NOISE', noise_scale, 'noise scale')
    noise_scale_w, text = get_label_value(text, 'NOISEW', noise_scale_w, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')
    
    return text, length_scale, noise_scale, noise_scale_w, cleaned

def get_label_value(text, label, default, warning_name='value'):
    """从文本中提取标签值"""
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            value = default
    else:
        value = default
    return value, text

def get_label(text, label):
    """从文本中提取布尔标签"""
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

def get_text(text, cleaned=False):
    """将文本转换为模型输入"""
    if hps_global is None:
        return None
    
    if cleaned:
        text_norm = text_to_sequence(text, hps_global.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps_global.symbols, hps_global.data.text_cleaners)
    
    if hps_global.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    
    text_norm = LongTensor(text_norm).to(device)
    return text_norm

def synthesize(text, speaker_id, length_scale, noise_scale, noise_scale_w, 
               model_path, config_path, output_path):
    """合成语音的主函数"""
    
    # 检查是否已加载模型
    if model_global is None or hps_global is None:
        speakers, load_result = load_model(model_path, config_path)
        if speakers is None:
            return None, load_result
    
    try:
        # 处理文本
        processed_text, length_scale, noise_scale, noise_scale_w, cleaned = process_text(
            text, length_scale, noise_scale, noise_scale_w
        )
        
        if processed_text is None or processed_text.strip() == "":
            return None, "请输入文本"
        
        # 转换为模型输入
        stn_tst = get_text(processed_text, cleaned=cleaned)
        if stn_tst is None:
            return None, "文本处理失败"
        
        # 推理
        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            
            audio = model_global.infer(
                x_tst, x_tst_lengths, sid=sid,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale
            )[0][0, 0].data.cpu().float().numpy()
        
        # 保存音频
        if not output_path:
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "vits_output.wav")
        
        write(output_path, hps_global.data.sampling_rate, audio)
        
        param_info = []
        if length_scale != 1.0:
            param_info.append(f"长度={length_scale:.2f}")
        if noise_scale != 0.667:
            param_info.append(f"噪声={noise_scale:.2f}")
        if noise_scale_w != 0.8:
            param_info.append(f"偏差={noise_scale_w:.2f}")
        
        param_str = f"（{', '.join(param_info)}）" if param_info else ""
        
        return output_path, f"✅ 合成成功！{param_str}\n音频已保存到：{output_path}"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 合成失败：{str(e)}"

def update_speaker_dropdown(model_path, config_path):
    """更新说话人下拉框"""
    speakers, result = load_model(model_path, config_path)
    if speakers:
        speaker_list = [(f"{name} (ID:{idx})", idx) for idx, name in enumerate(speakers)]
        return gr.Dropdown(choices=speaker_list, value=0), result
    else:
        return gr.Dropdown(choices=[("无说话人", 0)], value=0), result

def auto_load_default():
    """自动加载默认模型"""
    if os.path.exists(DEFAULT_MODEL_DIR):
        model_file, config_file = find_model_files(DEFAULT_MODEL_DIR)
        if model_file and config_file:
            speakers, result = load_model(model_file, config_file)
            if speakers:
                speaker_list = [(f"{name} (ID:{idx})", idx) for idx, name in enumerate(speakers)]
                return (
                    model_file, config_file,
                    gr.Dropdown(choices=speaker_list, value=0),
                    f"✅ 已自动加载默认模型：{os.path.basename(model_file)}"
                )
    return None, None, gr.Dropdown(choices=[("无说话人", 0)], value=0), "未找到默认模型"

# 创建Gradio界面
with gr.Blocks(title="VITS TTS GUI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 VITS 文本转语音 GUI
    
    基于VITS的文本合成语音界面，支持多种参数调节。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 模型配置区
            gr.Markdown("### 📁 模型配置")
            model_path = gr.File(
                label="选择VITS模型文件 (.pth)",
                file_types=[".pth"],
                type="filepath",
                value=DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else None
            )
            config_path = gr.File(
                label="选择配置文件 (.json)",
                file_types=[".json"],
                type="filepath",
                value=DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None
            )
            
            with gr.Row():
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                use_default_btn = gr.Button("📂 使用默认模型", variant="secondary")
            
            load_status = gr.Textbox(label="加载状态", interactive=False, lines=3)
            
            gr.Markdown("### 🎛️ 合成参数")
            length_scale = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="长度缩放 (LENGTH)",
                info="控制语速，值越大语速越慢"
            )
            noise_scale = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.667, step=0.1,
                label="噪声缩放 (NOISE)",
                info="控制随机性，值越大变化越大"
            )
            noise_scale_w = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                label="噪声偏差 (NOISEW)",
                info="控制音调变化"
            )
        
        with gr.Column(scale=2):
            # 输入输出区
            gr.Markdown("### 📝 文本输入")
            text_input = gr.Textbox(
                label="输入要合成的文本",
                placeholder="例如：[LENGTH=1.2][NOISE=0.5]你好，世界！",
                lines=5,
                value="你好，欢迎使用VITS语音合成！"
            )
            
            gr.Markdown("### 🗣️ 说话人选择")
            speaker_dropdown = gr.Dropdown(
                choices=[("请先加载模型", 0)],
                value=0,
                label="选择说话人ID",
                info="从下拉列表中选择说话人"
            )
            
            gr.Markdown("### 💾 输出设置")
            output_path = gr.Textbox(
                label="输出音频路径",
                placeholder="例如：output.wav（留空则使用临时文件）",
                value=""
            )
            
            synthesize_btn = gr.Button("🔊 合成语音", variant="primary", size="lg")
            
            with gr.Row():
                audio_output = gr.Audio(
                    label="合成结果",
                    type="filepath"
                )
            
            output_status = gr.Textbox(label="合成状态", interactive=False, lines=3)

    # 示例区
    gr.Markdown("### 📋 使用示例")
    
    examples = gr.Examples(
        examples=[
            ["[ZH]你好，世界！[ZH]", 1.0, 0.667, 0.8],
            ["[LENGTH=1.5][ZH]这是一个语速较慢的示例[ZH]", 1.5, 0.667, 0.8],
            ["[NOISE=1.2][ZH]这是一个随机性较强的示例[ZH]", 1.0, 1.2, 0.8],
            ["[LENGTH=0.8][NOISE=0.9][NOISEW=1.1][ZH]这是一个组合参数的示例ZH]", 0.8, 0.9, 1.1],
        ],
        inputs=[text_input, length_scale, noise_scale, noise_scale_w],
        label="点击示例快速填充"
    )
    
    gr.Markdown("""
    **输入格式说明：**
    - `[LENGTH=1.2]` - 设置语速（默认1.0）
    - `[NOISE=0.5]` - 设置噪声（默认0.667）
    - `[NOISEW=0.9]` - 设置噪声偏差（默认0.8）
    - `[CLEANED]` - 使用已清洗文本
    
    **完整示例：** `[LENGTH=1.2][NOISE=0.5][NOISEW=0.9][ZH]你好，欢迎使用VITS！[ZH]`
    **使用需知：** `输入时一定要记得在需要合成的文字两边加上语言标识符，例如[ZH]XXXXXX[ZH]`
    """)
    
    # 事件绑定
    load_btn.click(
        fn=update_speaker_dropdown,
        inputs=[model_path, config_path],
        outputs=[speaker_dropdown, load_status]
    )
    
    use_default_btn.click(
        fn=auto_load_default,
        inputs=[],
        outputs=[model_path, config_path, speaker_dropdown, load_status]
    )
    
    synthesize_btn.click(
        fn=synthesize,
        inputs=[
            text_input, speaker_dropdown,
            length_scale, noise_scale, noise_scale_w,
            model_path, config_path, output_path
        ],
        outputs=[audio_output, output_status]
    )
    
    # 启动时自动加载默认模型
    demo.load(
        fn=auto_load_default,
        inputs=[],
        outputs=[model_path, config_path, speaker_dropdown, load_status]
    )

# 启动界面
if __name__ == "__main__":
    # 检查默认模型路径
    if os.path.exists(DEFAULT_MODEL_DIR):
        print(f"默认模型目录：{DEFAULT_MODEL_DIR}")
        model_file, config_file = find_model_files(DEFAULT_MODEL_DIR)
        if model_file and config_file:
            print(f"找到默认模型：{os.path.basename(model_file)}")
            print(f"找到默认配置：{os.path.basename(config_file)}")
        else:
            print("默认目录中未找到模型文件")
    else:
        print(f"默认模型目录不存在：{DEFAULT_MODEL_DIR}")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        allowed_paths=['/kaggle/working/ema']
    )
