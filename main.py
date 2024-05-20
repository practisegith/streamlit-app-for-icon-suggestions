import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

user_tokenizer = None


model_name = "sentence-transformers/all-mpnet-base-v2"
sentence_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


icons = {
    "home": "C:\icon 2\icon images\homeicon.png",
    "search": "C:\icon 2\icon images\search icon.png",
    "chat":"C:\icon 2\icon images\icons8-chat-40.png",
    "upload":"C:\icon 2\icon images\icons8-upload-50.png",
    "download":"C:\icon 2\icon images\icons8-download-50.png",
    "pdf":"C:\icon 2\icon images\icons8-pdf-50.png",
    "mail":"C:\icon 2\icon images\icons8-mail-50.png",
    "database":"C:\icon 2\icon images\icons8-database-50.png",
    "notification":"C:\icon 2\icon images\icons8-notification-50.png",
    "bug":"C:\icon 2\icon images\icons8-bug-50.png"


}

def get_embedding(text, tokenizer):
   #simplifies the process by automatically converting the tokenized inputs into PyTorch tensors, making the code cleaner and more efficient
    encoded_input = tokenizer(text, return_tensors='pt')
    #Disables gradient calculation for efficiency during inference.
    with torch.no_grad():
        output = model(**encoded_input)#tokenized input prepared for feeding into the transformer model
    return output.pooler_output.squeeze(0)  #  extract and reshape the sentence embedding from the model's output

def main():
    st.title("Icon Selector with Sentence Embeddings")
    user_input = st.text_input("Enter an icon description:")

    if user_input:
        user_input = user_input.lower()


        user_embedding = get_embedding(user_input, user_tokenizer or sentence_tokenizer)#text input provided by the useri.e user_embeddings


        icon_embeddings = {}# used to store the embeddings of the icon descriptions,avoids recomputing these embeddings multiple times
        for description, icon_path in icons.items():
            if description not in icon_embeddings:
                icon_embeddings[description] = get_embedding(description, sentence_tokenizer)


        max_similarity = -float('inf')#finding the maximum value in a sequence
        closest_icon = None
        for description, icon_embedding in icon_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(user_embedding.unsqueeze(0), icon_embedding.unsqueeze(0))#cosine similarity similar two vectors are irrespective of their size
            if similarity > max_similarity:#similarity--> the similarity scores for individual icons,max_similarity--> maximum similarity between the user input and any of the icon descriptions
                max_similarity = similarity
                closest_icon = description

        if closest_icon:
            icon_path = icons[closest_icon]  # Use icon dictionary for path
            try:
                st.image(icon_path, caption=f"Closest Icon: {closest_icon}")
            except FileNotFoundError:
                st.error(f"Icon image not found: {icon_path}")
        else:
            st.write(f"No similar icon found for '{user_input}'.")

if __name__ == "__main__":
    main()

