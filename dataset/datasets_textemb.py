from typing import Dict, List
from torch.utils.data import Dataset
from datasets import load_dataset


class MSMarcoDataset(Dataset):
    """
    Dataset for MS MARCO triplet data
    FROM: sentence-transformers/msmarco-msmarco-MiniLM-L6-v3/triplet
    """

    def __init__(self, 
        dataset_path = "/mnt/tidal-alsh01/dataset/mmeb/text_embed/msmarco-msmarco-MiniLM-L6-v3/triplet",
        max_length = -1
    ) -> None:
        super(MSMarcoDataset, self).__init__()
        self.dataset = load_dataset(
            dataset_path,
            split="train"
        )
        self.max_length = max_length

    def __len__(self) -> int:
        if self.max_length >= 0:
            return self.max_length
        return len(self.dataset)

    def construct_messages(self, text):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}\nSummarize above sentence in one word: "}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"<emb>."}
                ]
            },
        ]
        return message

    def get_instance(self, index):
        item = self.dataset[index]
        query = item['query']
        positive = item['positive']
        negative = item['negative']
        
        message_query = self.construct_messages(query)
        message_positive = self.construct_messages(positive)
        message_negative = self.construct_messages(negative)

        return message_query, message_positive #, message_negative

    def __getitem__(self, i) -> Dict[str, List]:      
        return self.get_instance(i)

class HotpotQADataset(MSMarcoDataset):
    """
    Dataset for HotpotQA triplet data
    FROM: sentence-transformers/hotpotqa/triplet
    """

    def __init__(self, 
        dataset_path = "/mnt/tidal-alsh01/dataset/mmeb/text_embed/hotpotqa/triplet",
        max_length = -1
    ) -> None:
        super(HotpotQADataset, self).__init__(dataset_path, max_length)

    def get_instance(self, index):
        item = self.dataset[index]
        query = item['anchor']
        positive = item['positive']
        # negative = item['negative']
        
        message_query = self.construct_messages(query)
        message_positive = self.construct_messages(positive)
        # message_negative = self.construct_messages(negative)

        return message_query, message_positive #, message_negative

class NaturalQuestionsDataset(MSMarcoDataset):
    """
    Dataset for Natural Questions triplet data
    FROM: sentence-transformers/nq-distilbert-multilingual-nq-triplet
    """

    def __init__(self, 
        dataset_path = "/mnt/tidal-alsh01/dataset/mmeb/text_embed/natural-questions/pair",
        max_length = -1
    ) -> None:
        super(NaturalQuestionsDataset, self).__init__(dataset_path, max_length)
    
    def get_instance(self, index):
        item = self.dataset[index]
        query = item['query']
        positive = item['answer']
        
        message_query = self.construct_messages(query)
        message_positive = self.construct_messages(positive)

if __name__ == "__main__":
    print("Loading HotpotQA dataset...")
    dataset = HotpotQADataset()
    
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])

    # print("Loading MS MARCO dataset...")
    # dataset = MSMarcoDataset()
    
    # print(f"Dataset size: {len(dataset)}")
    # print(dataset[0])