#任务1文本生成
#！！！EleutherAI/gpt-neo-2.7B
from transformers import pipeline
generator = pipeline('text-generation',model='EleutherAI/gpt-neo-2.7B')
prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
         "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
         "researchers was the fact that the unicorns spoke perfect English."
res = generator(prompt,max_length=180)
print(res)

#！！！gpt-neo-1.3B
from happytransformer import HappyGeneration
happy_gen = HappyGeneration("GPT-NEO","EleutherAI/gpt-neo-1.3B")
result = happy_gen.generate_text("In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
         "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
         "researchers was the fact that the unicorns spoke perfect English.")
from happytransformer import GENSettings
args = GENSettings(no_repeat_ngram_size=2,max_length=150)
result1 = happy_gen.generate_text("In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
         "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
         "researchers was the fact that the unicorns spoke perfect English.", args=args)
print(result1.text)


#！！！Cerebras-GPT-1.3B
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-1.3B")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-1.3B")

text = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
          "researchers was the fact that the unicorns spoke perfect English."
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = pipe(text, max_length=180, do_sample=False, no_repeat_ngram_size=2)[0]
print(generated_text['generated_text'])

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, num_beams=5,
                        max_new_tokens=50, early_stopping=True,
                        no_repeat_ngram_size=2)
text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text_output[0])
#文本生成结果指标1
def diversity(text):
    words_list = text.strip().split()
    n_gram_1 = set(words_list)
    n_gram_2 = set(zip(words_list[:-1], words_list[1:]))
    n_gram_3 = set(zip(words_list[:-2], words_list[1:-1], words_list[2:]))
    return (len(n_gram_1) + len(n_gram_2) + len(n_gram_3)) / len(words_list)
文本生成结果指标2
import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def consistency(text, topic):
    # 使用BERT将文本和主题编码为向量
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    encoded_topic = tokenizer(topic, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_vec = model(**encoded_text)[0][:, 0, :]
        topic_vec = model(**encoded_topic)[0][:, 0, :]
    # 计算两个向量之间的余弦相似度
    similarity = torch.nn.functional.cosine_similarity(text_vec, topic_vec).item()
    return similarity
# 评价指标
a="The unicorn herd was discovered by a team of researchers from the University of California, Davis, and the National Geographic Society. The team, led by Dr. David R. Smith, a UC Davis professor of ecology and evolutionary biology, discovered the herd by following the tracks of the animals. They were able to follow the unicorn herd for more than a year, until they were forced to leave the valley. In the process, they discovered that there were more unicurs than previously thought.We were surprised to find that we were the only ones to have seen this herd,” said Dr Smith. “It was a very unusual discovery.Dr Smith and his team were also surprised by the"
print(diversity(a))
print(consistency(text,a))

#任务2实体检测
from transformers import pipeline
generator = pipeline('text-generation',model='EleutherAI/gpt-neo-1.3B')
generator = pipeline('text-generation',model='EleutherAI/gpt-neo-2.7B')
generator = pipeline('text-generation',model='cerebras/Cerebras-GPT-1.3B')
sample_data = '''
Entities: Person, Facility, Location, Organization, Work Of Art, Event, Date, Time, Nationality / Religious / Political group, Law Terms, Product, Percentage, Currency, Language, Quantity, Ordinal Number, Cardinal Number, Degree, Company, Food
Text: Google was incorporated as a privately held company on September 4, 1998 by Larry Page and Sergey Brin, while they were Ph.D. students at Stanford University in California. They own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock.
Entities Detected:
Google : Company
September 4, 1998 : Date
Larry Page : Person
Sergey Brin : Person
Stanford University : Organization
California : Location
14 percent : Percentage
56 percent : Percentage
Ph. D. : Degree

Entities: Person, Facility, Location, Organization, Work Of Art, Event, Date, Time, Nationality / Religious / Political group, Law Terms, Product, Percentage, Currency, Language, Quantity, Ordinal Number, Cardinal Number, Degree, Company, Food
Text: The U.S. President Donald Trump came to visit Ahmedabad for the first time at Reliance University with our Prime Minister Narendra Modi in February 2020.
Entities Detected:
U.S. : Location
Donald Trump : Person
Ahmedabad : Location
Narendra Modi : Person
Reliance University : Organization
February 2020 : Date

Entities: Person, Facility, Location, Organization, Work Of Art, Event, Date, Time, Nationality / Religious / Political group, Law Terms, Product, Percentage, Currency, Language, Quantity, Ordinal Number, Cardinal Number, Degree, Company, Food
Text: SpaceX is an aerospace manufacturer and space transport services company headquartered in California. It was founded in 2002 by entrepreneur and investor Elon Musk to reduce space transportation costs and enable Mars's colonizationwith the goal of reducing space transportation costs and enabling the colonization of Mars. Elon Musk is an American Entrepreneur.
Entities Detected:
SpaceX : Company
California : Location
2002 : Date
Elon Musk : Person
Mars : Location
American : Nationality / Religious / Political group
'''
input_text = '''Entities: Person, Facility, Location, Organization, Work Of Art, Event, Date, Time, Nationality / Religious / Political group, Law Terms, Product, Percentage, Currency, Language, Quantity, Ordinal Number, Cardinal Number, Degree, Company, Food
Text: Amazon.com, Inc., known as Amazon, is an American online business and cloud computing company. It was founded on July 5, 1994 by Jeff Bezos. It is based in Seattle, Washington. It is the largest Internet-based store in the world by total sales and market capitalization. Amazon.com started as an online bookstore. When it got bigger, it started selling DVDs, Blu-rays, CDs, video downloads/streaming, MP3s, audiobooks, software, video games, electronics, apparel, furniture, food, toys, and jewelry. It also makes consumer electronics like Kindle e-readers, Fire tablets, Fire TV, and Echo. It is the world's largest provider of cloud computing services. Amazon also sells products like USB cables using the name AmazonBasics.'''

prompt_text = sample_data + '\n\n' + input_text
text = generator(prompt_text , do_sample=True, max_length=1000)
print(text[0]['generated_text'])

#任务3意图分类
from transformers import pipeline
# generator = pipeline('text-generation',model='EleutherAI/gpt-neo-1.3B')
generator = pipeline('text-generation',model='EleutherAI/gpt-neo-2.7B')
# generator = pipeline('text-generation',model='cerebras/Cerebras-GPT-1.3B')
prompt_text = '''

Sentence: listen to westbam alumb allergic on Google music
Classification: PlayMusic

Sentence: Give me a list of movie times for films in the area
Classification: SearchScreeningEvent

Sentence: Show me the picture creatures of light and darkness
Classification: SearchCreativeWork

Sentence: I would like to go to the popular bistro in oh
Classification: BookRestaurant'''

input_text = "Sentence: What is the weather like in the city of frewen in the country of"

prompt_text = prompt_text + "\n" + input_text
text = generator(prompt_text , do_sample=True, max_length=150)

print(text[0]['generated_text'])