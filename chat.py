import random
import json
import os

import torch


from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import nltk
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Matsukiyo AI"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                if tag == "order":
                    medicines = {
                        "Healing Potion" : 100,
                        "Stamina Potion" : 120,
                        "Mana Potion" : 110,
                        "Poison Potion" : 90,
                        "Stun Resist Potion" : 90
                    }

                    prompt = input("Yes! Do you want to preorder? [Y/n] ")
                    if prompt == "Y" or "y":
                        os.system("cls")
                        order_list = []
                        address = ""
                        price = 0
                        switch = 1
                        
                        while switch == 1:
                            print("Matsukiyo AI: Here are the available medicines at the moment \n [0] Healing Potion - 100yen\n","[1] Stamina Potion - 120yen\n", "[2] Mana Potion - 110yen\n", "[3] Poison Resist Potion - 90yen\n", "[4] Stun Resist Potion - 90yen\n")
                            order = int(input("You: "))

                            order_list.append(list(medicines)[order])

                            price = price + list(medicines.values())[order]
                            print("Matsukiyo AI: ", list(medicines)[order], " is now added to your cart!\n", "Matsukiyo AI: Would you like to add another drink?")
                            
                            prompt2 = input("You: ")

                            if prompt2 == "yes":
                                switch = 1
                            else:
                                switch = 0
                                os.system("cls")

                    print("Matsukiyo AI: Can I have your complete delivery address?")
                    del_address = str(input("You: "))
                    address = del_address

                    print("Do you want to checkout your order? [Y/n]")
                    promt3 = str(input("You: "))
                    
                    if promt3.lower() == "y":
                        print("Matsukiyo AI: Here's the list of your orders: \n")
                        for meds in order_list:
                            print("- ", meds)
                        print("It will be delivered to:", address)

                        print("\n Matsukiyo AI: Kindly pay the amount of: ", price, "php "," upon pick up! ^^")
                    elif promt3.lower() == "n":
                        print('Thank you for using our website. Feel free to visit again.')
    else:
        print(f"{bot_name}: I do not understand..")