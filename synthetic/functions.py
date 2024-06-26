import numpy as np
import random
import itertools
import functools


class BaseFunction:
    """List of functions applied on data"""
    @staticmethod
    def identity(xstr):
        """Identify function"""
        return xstr

    @staticmethod
    def map(xstr, mapping):
        """Apply bijection to tokens"""
        return mapping[xstr]

    @staticmethod
    def permute(xstr, mapping):
        """Permute the token"""
        return xstr[mapping]


class CreateFunctions:
    """Generate a family of functions and compose them together"""
    def __init__(self, cfg):
        self.n_alphabets = cfg.n_alphabets
        self.seq_len = cfg.seq_len
        self.function_properties = cfg.function

    def generate_bijections(self):
        """
        Create a set of bijective mapping functions. 

        Args:
            nfuncs: number of different map functions
        """
        n_functions = self.function_properties.n_functions
        depth = self.function_properties.depth

        if self.function_properties.permute:
            all_functions = []
            if not self.function_properties.repeat:
                for d in range(depth):
                    ln = self.n_alphabets if d!=0 else self.seq_len

                    functions = [np.arange(ln)]
                    for i in range(n_functions):
                        functions.append((np.random.permutation(ln)))
                    all_functions.append(functions)

        else:
            all_functions = []
            if not self.function_properties.repeat:
                for d in range(depth):
                    functions = [np.arange(self.n_alphabets)]
                    for i in range(n_functions):
                        functions.append((np.random.permutation(self.n_alphabets)))
                    all_functions.append(functions)

            else:
                functions = [np.arange(self.n_alphabets)]
                for i in range(n_functions):
                    functions.append((np.random.permutation(self.n_alphabets)))
                for d in range(depth):
                    all_functions.append(list(functions))

        return all_functions

    def reduce_functions(self, fn_list):
        depth = self.function_properties.depth
        cur_fn = np.arange(self.n_alphabets)

        for i in range(depth):
            cur_fn = fn_list[i][cur_fn]
        return cur_fn

    def compose_bijections(self):
        depth = self.function_properties.depth
        n_functions = self.function_properties.n_functions
        all_functions = self.generate_bijections()

        function_info = {
            'functions': all_functions,
            'task_id': [],
            'composition_reduced': []
        }
        composed_functions =[]

        for idx in itertools.product(range(n_functions+1), repeat=depth):

            fn_list = [all_functions[d][i] for d, i in enumerate(idx)]

            if not self.function_properties.permute:
                reduced_func = self.reduce_functions(fn_list)
                fn = functools.partial(BaseFunction.map,
                                       mapping=reduced_func)
            else:
                reduced_func = None
                fn = None

            fnmap = [BaseFunction.map for d in range(depth)]
            if self.function_properties.permute:
                fnmap[0] = BaseFunction.permute

            fnpartial_list = [functools.partial(fnmap[d], mapping=fn_list[d]) for d in range(depth)]

            composed_functions.append((idx, fn, fnpartial_list))
            function_info['task_id'].append(idx)
            function_info['composition_reduced'].append(reduced_func)


        reduced_functions = np.array(function_info['composition_reduced'])
        if not self.function_properties.permute:
            print("Number of unique/total functions: %d/%d" %  \
                  (len(np.unique(reduced_functions, axis=0)),
                  len(reduced_functions)))

            for key in function_info:
                function_info[key] = np.array(function_info[key])

        return composed_functions, function_info

    def get_train_functions(self, composed_functions):

        depth = self.function_properties.depth
        n_functions = self.function_properties.n_functions

        alltask_ids = set(itertools.product(range(n_functions+1),
                                            repeat=depth))

        if self.function_properties.split.strategy == "base":

            # Base tasks
            base_ids = [tuple(np.zeros(depth, dtype=int))]
            for d in range(depth):
                for i in range(1, n_functions+1):
                    task_id = np.zeros(depth, dtype=int)
                    task_id[d] = i
                    base_ids.append(tuple(task_id))

            base_ids = set(base_ids)
            remaining_tasks = alltask_ids - base_ids

            additional_tasks = random.sample(
                remaining_tasks,
                self.function_properties.split.n_compositions)

            traintask_ids = list(base_ids) + list(additional_tasks)

            print("Number of base  tasks: %d" % len(base_ids))
            print("Number of train tasks: %d" % len(traintask_ids))

        elif self.function_properties.split.strategy == "random":

            traintask_ids = random.sample(
                alltask_ids,
                self.function_properties.split.n_compositions)

        elif self.function_properties.split.strategy == "random_biased":
            n_identity = self.function_properties.split.n_identity

            sub_taskids = []
            for tid in alltask_ids:
                if np.sum(np.array(tid) == 0) == n_identity:
                    sub_taskids.append(tid)

            maxlen = len(sub_taskids)
            if self.function_properties.split.n_compositions > maxlen:
                raise ValueError

            traintask_ids = random.sample(
                sub_taskids,
                self.function_properties.split.n_compositions)

            print("Number of possible functions: %d" % len(sub_taskids))
            print("Number of train tasks: %d" % len(traintask_ids))

        elif self.function_properties.split.strategy == "randombase_combo":
            # base functions
            base_ids = [tuple([(d, 0) for d in range(depth)])]
            for d in range(depth):
                for i in range(1, n_functions+1):
                    task_id = np.zeros(depth, dtype=int)
                    task_id = [(k, 0) for k in range(depth)]
                    task_id[d] = (d, i)
                    base_ids.append(tuple(task_id))

            nf_choices = []
            for d in range(depth):
                nf_choices.append([(d, i) for i in range(n_functions)])
            all_tids = list(itertools.product(*nf_choices))
            inorder_tasks = random.sample(
                all_tids,
                self.function_properties.split.n_compositions_inorder)

            # out-of-order functions
            nf_choices = []
            for d in range(depth):
                nf_list = []
                for d2 in range(depth):
                    if d2 != d:
                        nf_list += [(d2, i) for i in range(n_functions)]
                nf_choices.append(nf_list)

            all_tids = list(itertools.product(*nf_choices))
            additional_tasks = random.sample(
                all_tids,
                self.function_properties.split.n_compositions)
            traintask_ids = list(base_ids) + list(additional_tasks) + list(inorder_tasks)

        elif self.function_properties.split.strategy == "random_combo":
            base_ids = [tuple([(d, 0) for d in range(depth)])]

            nf_choices = []
            for d in range(depth):
                nf_list = []
                for d2 in range(depth):
                    if d2 != d:
                        nf_list += [(d2, i) for i in range(n_functions)]
                nf_choices.append(nf_list)

            all_tids = list(itertools.product(*nf_choices))
            additional_tasks = random.sample(
                all_tids,
                self.function_properties.split.n_compositions)
            traintask_ids = list(base_ids) + list(additional_tasks)

        elif self.function_properties.split.strategy == "base_combo":
            base_ids = [tuple([(d, 0) for d in range(depth)])]
            for d in range(depth):
                for i in range(1, n_functions+1):
                    task_id = np.zeros(depth, dtype=int)
                    task_id = [(k, 0) for k in range(depth)]
                    task_id[d] = (d, i)
                    base_ids.append(tuple(task_id))

            nf_choices = []
            for d in range(depth):

                nf_list = []
                for d2 in range(depth):
                    if d2 != d:
                        nf_list += [(d2, i) for i in range(n_functions)]
                nf_choices.append(nf_list)

            all_tids = list(itertools.product(*nf_choices))
            additional_tasks = random.sample(
                all_tids,
                self.function_properties.split.n_compositions)
            traintask_ids = list(base_ids) + list(additional_tasks)


            print("Number of train tasks: %d" % len(traintask_ids))

        train_fns = []

        if "combo" not in self.function_properties.split.strategy:
            for idx, fn in enumerate(composed_functions):
                if fn[0] in traintask_ids:
                    train_fns.append(fn)
        else:
            train_fns = []

            for tid  in traintask_ids:
                fnpartial_list = [functools.partial(
                    BaseFunction.map, mapping=self.finfo['functions'][x1][x2]) for x1, x2 in tid]
                train_fns.append((tid, None, fnpartial_list))


        return train_fns, traintask_ids

    def compose(self):
        """
        Compose together different sets of functions. Divide them into 2 groups,
        _train_ and  _all_. _train_ is seen during training, _all is the set of
        all compositions of functions. Not that all compositions are not seen
        during training.
        """
        composed_functions = {'train': [], 'all': []}

        allcomp_functions, info = self.compose_bijections()
        self.finfo = info
        train_functions, train_ids = self.get_train_functions(allcomp_functions)

        composed_functions['all'] = allcomp_functions
        composed_functions['train'] = train_functions

        if self.function_properties.split.strategy != "base_combo":
            info['train_id'] = np.array(train_ids)
        else:
            info['train_id'] = train_ids

        return composed_functions, info
