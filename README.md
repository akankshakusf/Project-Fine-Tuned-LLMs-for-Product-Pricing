## Project-Fine-Tuned-LLAMA- LLMs-for-Product-Pricing

Dataset that I have Fine Tuning is Hugging Face : https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
The Dataset on hugging face has lots of categories (Number of Appliances: 94,327) and hence based on my resources that can be handled by Google collab i will use only a few categories.
link of categories : https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories

Steps to go about digging into the data at hand:
1. Investigate the dataset
2. Parse the data into objects
3. Visualize the data. Understand the skewness of the data.
4. Access the quality of the data at deeper level
5. Curate the data of you liking that is eg. take the quarters of the data has less missing values.
6. Save the data to hugging face finally that will used for training.

After importing the data from hugging face we tokenize the data with help of Qwen Model. "Qwen/Qwen2.5-1.5B-Instruct"
---
At the time of the training of the model we are passing prompt like this example.
* Pay attention training data is like this that has price value. But the Test Prompt will not have the price value.
* I am only taking a max of 180 tokens

## Now it's time to curate our dataset

We select items that cost between 1 and 999 USD

We will be create Item instances, which truncate the text to fit within 180 tokens using the right Tokenizer 
I will be making use of Hugging Face Tokenizer as OpenAIEmbeddings and all can get quite expensive.

And will create a prompt to be used during Training.

Items will be rejected if they don't have sufficient characters.

## But why 180 tokens??

A student asked me a great question - why are we truncating to 180 tokens? How did we determine that number? (Thank you Moataz A. for the excellent question).

The answer: this is an example of a "hyper-parameter". In other words, it's basically trial and error! We want a sufficiently large number of tokens so that we have enough useful information to gauge the price. But we also want to keep the number low so that we can train efficiently. You'll see this in action in Week 7.

I started with a number that seemed reasonable, and experimented with a few variations before settling on 180. If you have time, you should do the same! You might find that you can beat my results by finding a better balance. This kind of trial-and-error might sound a bit unsatisfactory, but it's a crucial part of the data science R&D process.

There's another interesting reason why we might favor a lower number of tokens in the training data. When we eventually get to use our model at inference time, we'll want to provide new products and have it estimate a price. And we'll be using short descriptions of products - like 1-2 sentences. For best performance, we should size our training data to be similar to the inputs we will provide at inference time.

## But I see in items.py it constrains inputs to 160 tokens?

Another great question from Moataz A.! The description of the products is limited to 160 tokens because we add some more text before and after the description to turn it into a prompt. That brings it to around 180 tokens in total.



## Training data prompt example
How much does this cost to the nearest dollar?

Refrigerator Water Inlet Valve 1/4 Inlet Fitting Icemaker Water Inlet Valve with Guard Replaces
❄❄PART DESCRIPTION single outlet valve with guard provides water for ice maker and water dispenser. refrigerator water valve has 1/4 inlet connector and new quick connect outlet connector ❄❄REPLACE PARTS NUMBERS can directly replace the following models MV469, (Please make sure your model is correct before placing an order, if you are unsure, you can contact us at any time) ❄❄EASY INSTALLATION water inlet valve can solve the following problems Ice maker leaked.; the dispenser does not work; the ice maker does not make ice. Tools for the installation process require a screwdriver and wrench, no additional screws are required, allowing

Price is $19.00

# Testing data prompt example
How much does this cost to the nearest dollar?

Fuxury 12 Inch Impulse Bag Sealer, Heat Sealer Machine Closer for Cookies, Manual Hot Min Sealing Machine for Poly Bag & Shrink Wrap with 2 Free Repair Kit (Blue)
Professional Heat Sealer Fuxury impulse bag sealer precision electronic control circuit to controls the time and the heat automatically.The heat sealer compact size 12 length, 2mm sealing width, metal quality and double paint, stability and good rust effect. Bag sealer equipment includes 2 replacement kits. Convenient Use Fuxury bag sealer are safe and easy to use. The heating element in the impulse sealer will not continue to heat up. Once pressure is applied to the sealing arm, the heat sealer will quickly seal the material and form a seal. The

Price is $
---

## All the datasets that i am loading.
dataset_names = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]
## So, I am wrking with the dataset of 2,811,408 datapoints of all the categories


