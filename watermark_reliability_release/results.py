
#%%
from PIL import Image, ImageDraw, ImageFont
import re
from IPython.display import display
from matplotlib import pyplot as plt
from utils.io import load_jsonlines
from datasets import Dataset
# %%
# ppl_file = load_jsonlines(
#     'run_outputs/opt1.3b_N500_T200_eval/gen_table_w_metrics.jsonl')
folder="mistral_instruct_shorter_2"
z_file = load_jsonlines(
    f"run_outputs/{folder}/gen_table_w_metrics copy.jsonl")
ppl_file = load_jsonlines(
    f"run_outputs/{folder}/gen_table_w_metrics.jsonl")
psp_file = load_jsonlines(
    f"run_outputs/{folder}/gen_table_w_metrics_psp.jsonl")
    # 'baseline/gen_table_w_metrics.jsonl')
# ppl_file = Dataset.from_list(ppl_file)
z_file = Dataset.from_list(z_file)
ppl_file = Dataset.from_list(ppl_file)
psp_file = Dataset.from_list(psp_file)
z_file=z_file.add_column('w_wm_output_attacked_ppl',ppl_file['w_wm_output_attacked_ppl'])
z_file=z_file.add_column('w_wm_output_attacked_baseline_ppl',ppl_file['w_wm_output_attacked_baseline_ppl'])
z_file=z_file.add_column('w_wm_output_vs_w_wm_output_attacked_p_sp',psp_file['w_wm_output_vs_w_wm_output_attacked_p_sp'])
z_file=z_file.add_column('w_wm_output_vs_w_wm_output_attacked_baseline_p_sp',psp_file['w_wm_output_vs_w_wm_output_attacked_baseline_p_sp'])

# ppl_file = ppl_file.filter(lambda x: x['w_wm_output_length'] > 10)
z_file = z_file.filter(lambda x: x['w_wm_output_length'] > 10)
z_file = z_file.filter(lambda x: x['w_wm_output_length'] <400)

# %%
cols = ["no_wm_output_z_score", "w_wm_output_z_score", "w_wm_output_p_value",
         "w_wm_output_winmax-1_z_score", "w_wm_output_ppl", "w_wm_output_attacked_ppl", "w_wm_output_attacked_baseline_ppl", "w_wm_output_vs_w_wm_output_attacked_p_sp", "w_wm_output_vs_w_wm_output_attacked_baseline_p_sp", "w_wm_output_attacked_log_diversity", "w_wm_output_attacked_baseline_log_diversity", "w_wm_output_log_diversity", "w_wm_output_attacked_winmax-1_z_score", "w_wm_output_attacked_baseline_winmax-1_z_score", "w_wm_output_attacked_z_score", "w_wm_output_attacked_p_value", "w_wm_output_attacked_baseline_z_score", "w_wm_output_attacked_baseline_p_value", "no_wm_output_winmax-1_z_score",
        ]

for col in cols:
    if col in z_file.features:
        print(col, sum(z_file[col])/len(z_file),"max",max(z_file[col]),"min",min(z_file[col]))
        
# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
# %%

# Define the color codes
color_codes = {
    '\033[92m': (0, 150, 0),  # Green
    '\033[91m': (200, 0, 0),  # Red
    '\033[0m': (0, 0, 0)      # Reset to black
}


class TextDrawer:
    def __init__(self, width, height, font_path):
        self.img = Image.new('RGB', (width, height), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.font = ImageFont.truetype(font_path, 30)
        self.x, self.y = 10, 10
        self.line_height=None
        self.cache={"word":[],"length":[]}

    def clear_cache(self):
        word_width=sum(self.cache["length"])
        if self.x + word_width > self.img.width:  # If the word goes beyond the image width
            self.x = 10  # Reset the x-coordinate
            self.y += self.line_height  # Move to the next line
        for ((word,color),l) in zip(self.cache["word"],self.cache["length"]):
            self.d.text((self.x, self.y), word,
                        font=self.font, fill=color)
            self.x += l  # Move the x-coordinate to the right for the next word
        self.cache={"word":[],"length":[]}
    
    def draw_colored_text(self, text):
        # Split the text into segments based on the color codes
        segments = re.split('(\033\[\d+m)', text)

        # Initialize variables
        color = (0, 0, 0)  # Default color is black

        # Draw each segment with its color
        for segment in segments:
            if segment in color_codes:
                color = color_codes[segment]  # Change the color
            else:
                words = re.split(r'(?<= )', segment)
                for word in words:
                    bbox = self.d.textbbox(
                        (self.x, self.y), text=word, font=self.font)
                    word_width = bbox[2] - bbox[0]
                    if not self.line_height:
                        self.line_height = bbox[3] - bbox[1]+10
                    
                    self.cache["word"].append((word,color))
                    self.cache["length"].append(word_width)
                    if word.endswith(" "):
                        self.clear_cache()

        self.clear_cache()       
        self.y += self.line_height
        self.x=10

    def display(self):
        # Display the image
        display(self.img)


# %%
# get the first 5 rows of the dataset

def show_color(sentence, mask):
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    token_ids = tokenizer.encode(sentence, return_tensors='pt', add_special_tokens=False)[0]
    sentence=""
    token=[]
    for i in range(len(token_ids)):
        i+=1
        new_word_sentence=tokenizer.decode(token_ids[:i])
        len_old_sentence=len(sentence)
        token.append(new_word_sentence[len_old_sentence:])
        sentence=new_word_sentence
        
        
    # token = [tokenizer.decode([x]) for x in token_ids]
    print(len(token), len(mask))
    str="".join(token[:4])
    token=token[4:]
    for i in range(len(token)):
        if mask[i]:
            str+=GREEN + token[i] + RESET
        else:
            str+=RED + token[i] + RESET
    str+=""
    return str
#%%
import torch
good=z_file.filter(lambda x: abs(x['w_wm_output_length']-x["w_wm_output_attacked_length"] ) < 20)
good=good.filter(lambda x: abs(x['w_wm_output_length']-x["w_wm_output_attacked_baseline_length"] ) < 20)
difference_in_z=zip(good['w_wm_output_z_score'],good['w_wm_output_attacked_z_score'],good['w_wm_output_attacked_baseline_z_score'])
difference_in_z=[x[0]-x[1]+2*(x[2]-x[1]) for x in difference_in_z]
if 'difference_in_z' in good.features:
    good=good.remove_columns(['difference_in_z'])
good=good.add_column('difference_in_z', difference_in_z)
good = good.filter(
    lambda x: torch.tensor(x['w_wm_output_attacked_green_token_mask']).sum()>len(x['w_wm_output_attacked_green_token_mask'])*0.25 >= 0.5)
# %%

good=good.sort('difference_in_z')
for i, item in enumerate(good.to_list()[-5:-5]):
    td=TextDrawer(900, 1000, 'times.ttf')

    orig = item['no_wm_output']
    wm = item['w_wm_output']
    atk = item['w_wm_output_attacked']
    atkb = item['w_wm_output_attacked_baseline']
    msk=torch.tensor(item['w_wm_output_attacked_green_token_mask'])
    print(msk.sum()/len(msk))
    # no_wm_output_green_token_mask
    # orig_id=tokenizer.encode(orig, return_tensors='pt', add_special_tokens=False)[0]
    # print("Original:")
    # show_color(orig, item['no_wm_output_green_token_mask'])
    td.draw_colored_text("Watermarked:")
    td.draw_colored_text(show_color(wm, item['w_wm_output_green_token_mask']))
    td.draw_colored_text("Attacked with BOW:")
    td.draw_colored_text(show_color(atk, item['w_wm_output_attacked_green_token_mask']))
    td.draw_colored_text("Baseline Attack:")
    td.draw_colored_text(show_color(atkb, item['w_wm_output_attacked_baseline_green_token_mask']))
    td.display()
    

# %%

import numpy as np    

def draw_fit(x_col,y_col,ax,color='orange'):
    x_data = z_file[x_col] 
    y_data = z_file[y_col] 

    # 使用numpy的polyfit函数来拟合这些数据点
    # 第三个参数是多项式的阶数，例如，2表示二次多项式
    coefficients = np.polyfit(x_data, y_data, 3)

    # 使用拟合的参数来创建一个表示拟合曲线的函数
    polynomial = np.poly1d(coefficients)

    # 生成一些x值
    x_fit = np.linspace(min(x_data), max(x_data), 100)

    # 计算对应的y值
    y_fit = polynomial(x_fit)
    ax.plot(x_fit, y_fit,color=color,linewidth=4)

# 绘制原始数据点


# %%

# %%
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from math import log1p

# 1*2 canvas
# set default font size
plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] 
fig, axs = plt.subplots(2, 1, figsize=(20, 20))
ax1,ax2=axs
fig.subplots_adjust(hspace=0.3)  # Increase the margin between ax1 and ax2
cmap=plt.get_cmap('viridis_r')
norm=LogNorm(vmin=1,vmax=50)  # 0-50 ppl

sm=ScalarMappable(cmap=cmap,norm=norm)

sm.set_array([])
def transparent(color, alpha=0.1):
    return (color[0], color[1], color[2], alpha)

for i,ex in tqdm(enumerate(z_file)):
    orig_z = ex['w_wm_output_z_score']
    attacked_z = ex['w_wm_output_attacked_z_score']
    orig_len = ex['w_wm_output_length']
    attacked_len = ex['w_wm_output_attacked_length']
    baseline_z = ex['w_wm_output_attacked_baseline_z_score']
    baseline_len = ex['w_wm_output_attacked_baseline_length']
    atk_ppl=ex['w_wm_output_attacked_ppl']
    bsl_ppl=ex['w_wm_output_attacked_baseline_ppl']

    img1=ax1.plot([orig_len, attacked_len], [orig_z, attacked_z],
             'o-', color=transparent(cmap(norm(atk_ppl))))
    img2=ax2.plot([orig_len, baseline_len], [orig_z, baseline_z],
             'o-', color=transparent(cmap(norm(bsl_ppl))))
    
    # if i>100:
    #     break

atk_psp=torch.tensor(psp_file['w_wm_output_vs_w_wm_output_attacked_p_sp']).mean()
bsl_psp=torch.tensor(psp_file['w_wm_output_vs_w_wm_output_attacked_baseline_p_sp']).mean()

ax1.set_xlabel('Length')
ax1.set_ylabel('z-score')
ax1.set_title(f'BoW Paraphrase, avg P-SP = {atk_psp:.3f}', fontsize=40)
ax1.grid(True)
ax2.set_xlabel('Length')
ax2.set_ylabel('z-score')
ax2.set_title(f'Baseline Paraphrase, avg P-SP = {bsl_psp:.3f}', fontsize=40)
ax2.grid(True)
# add colorbar
cbar=plt.colorbar(sm, ax=axs.ravel().tolist())
cbar.set_label('Perplexity')
# label for 1,3,10,30
label=torch.tensor([1,2,3,5,10,20,30,50])
cbar.set_ticks(label.tolist())
cbar.set_ticklabels(label.tolist())

# suptitle
# plt.tight_layout()
plt.suptitle("3-Way Beam Search",fontsize=50)

draw_fit('w_wm_output_length','w_wm_output_z_score',ax1)
draw_fit('w_wm_output_attacked_length','w_wm_output_attacked_z_score',ax1)
draw_fit('w_wm_output_attacked_baseline_length','w_wm_output_attacked_baseline_z_score',ax2)
draw_fit('w_wm_output_length','w_wm_output_z_score',ax2)
# ax1t=ax1.twinx()
# ax2t=ax2.twinx()
# ax1t.set_ylabel('P-SP score')
# ax2t.set_ylabel('P-SP score')
# draw_fit('w_wm_output_length',
#          'w_wm_output_vs_w_wm_output_attacked_p_sp', ax1t,color=(0.8,0,0))
# draw_fit('w_wm_output_length',
#             'w_wm_output_vs_w_wm_output_attacked_baseline_p_sp', ax2.twinx(),color=(0.8,0,0))


# %%
