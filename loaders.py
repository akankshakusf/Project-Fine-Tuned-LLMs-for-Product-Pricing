from datetime import datetime  # to track how long the script takes
from tqdm import tqdm  # shows progress bars during long operations
from datasets import load_dataset  # used to load datasets from Hugging Face
from concurrent.futures import ProcessPoolExecutor  # for parallel processing
from items import Item  # custom class that handles filtering and inclusion logic for products

# Constants
CHUNK_SIZE = 1000  # process 1000 products at a time
MIN_PRICE = 0.5  # filter out items cheaper than this
MAX_PRICE = 999.49  # filter out items more expensive than this

# Main class to handle item loading and processing
class ItemLoader:
    def __init__(self, name):
        self.name = name  # category name, like "toys" or "electronics"
        self.dataset = None  # placeholder for the dataset we'll load

    def from_datapoint(self, datapoint):
        """
        Try to create an Item object from a single datapoint (i.e., one product)
        If the price is valid and within range, return the Item
        Otherwise, return None
        """
        try:
            price_str = datapoint['price']  # get the price as a string
            if price_str:  # make sure it's not empty
                price = float(price_str)  # convert string to float
                if MIN_PRICE <= price <= MAX_PRICE:  # only accept prices within limits
                    item = Item(datapoint, price)  # wrap it in an Item class
                    return item if item.include else None  # include only if it's marked as valid
        except ValueError:
            return None  # ignore invalid prices that can't be converted to float

    def from_chunk(self, chunk):
        """
        Go through a chunk of products and return a cleaned-up list of Items
        """
        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)  # validate and convert datapoint to Item
            if result:
                batch.append(result)  # add valid items only
        return batch

    def chunk_generator(self):
        """
        Break the full dataset into smaller 1000-product chunks
        to avoid memory overload and support parallel processing
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))  # yield chunk

    def load_in_parallel(self, workers):
        """
        Process chunks in parallel using multiple CPU cores.
        This makes it much faster for big datasets.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1  # total number of chunks
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map each chunk to a worker, show progress bar
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)  # merge results from all workers

        # assign the category name to each item
        for result in results:
            result.category = self.name
        return results

    def load(self, workers=8):
        """
        Load the Amazon category dataset and process it using parallel workers.
        This is the main function you’ll call.
        """
        start = datetime.now()  # mark start time
        print(f"Loading dataset {self.name}", flush=True)
        
        # Load dataset from Hugging Face: e.g., "raw_meta_toys"
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                                    f"raw_meta_{self.name}", 
                                    split="full", 
                                    trust_remote_code=True)

        results = self.load_in_parallel(workers)  # do the processing
        finish = datetime.now()  # mark end time

        print(f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins", flush=True)
        return results  # return the final list of cleaned items
