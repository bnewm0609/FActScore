import pickle
import os
import time

from filelock import FileLock

class LM(object):

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_lock = FileLock(self.cache_file + ".lock")
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()

        # if prompt.endswith(" True or False?\nAnswer:"):
        if prompt.endswith(" True or False?\nOutput:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated
    
    def generate_batch(self, prompts, sample_idx=0, max_sequence_length=2048, max_output_length=128, batch_size=32):
        # breakpoint()
        prompts = [prompt.strip() for prompt in prompts]
        if self.model is None:
            self.load_model()
        
        prompts_uncached = [prompt for prompt in prompts if f"{prompt}_{sample_idx}" not in self.cache_dict]
        
        generated_uncached = []
        # scores_uncached = []
        for i in range(0, len(prompts_uncached), batch_size):
            prompts_batch = prompts_uncached[i: i+batch_size]
            # if prompts[0].endswith(" True or False?\nAnswer:"): # The prompts say "Output" and not "Answer" so this doesn't work
            if prompts[0].endswith(" True or False?\nOutput:"): # The prompts say "Output" and not "Answer" so this doesn't work
                gens, scores = self._generate(prompts_batch, max_sequence_length=max_sequence_length, max_output_length=1)
            else:
                gens, scores = self._generate(prompts_batch, max_sequence_length=max_sequence_length, max_output_length=max_output_length)
            generated_uncached.extend(list(zip(gens, scores)))
            # scores_uncached.extend(scores)
        
        for prompt, generation in zip(prompts_uncached, generated_uncached):
            self.cache_dict[f"{prompt}_{sample_idx}"] = generation
        self.add_n += len(prompts)
        
        generated = []
        # scores = []
        for prompt in prompts:
            output = self.cache_dict[f"{prompt}_{sample_idx}"]
            generated.append(output)
            # if isinstance(output, tuple):
            #     generated.append(output[0])
            # else:
            #     generated.append(output)
        # generated = [self.cache_dict[f"{prompt}_{sample_idx}"] for prompt in prompts]
        return generated
        

    def save_cache(self):
        if self.add_n == 0:
            return

        with self.cache_lock:
            # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
            for k, v in self.load_cache().items():
                self.cache_dict[k] = v

            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with self.cache_lock:
                        with open(self.cache_file, "rb") as f:
                            cache = pickle.load(f)
                        break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache



