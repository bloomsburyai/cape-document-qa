import re
import pickle
from os.path import join, exists, relpath
from os import walk
from cape_document_qa.cape_config import CORPUS_DIR
from docqa.utils import flatten_iterable
from docqa.triviaqa import evidence_corpus


class PatchedEvidenceCorpusTxt(object):
    """
    Corpus of the tokenized text from the given TriviaQa evidence documents.
    Allows the text to be retrieved by document id
    """

    _split_all = re.compile("[\n ]")
    _split_para = re.compile("\n\n+")  # FIXME we should not have saved document w/extra spaces...

    def __init__(self, file_id_map=None):
        self.directory = join(CORPUS_DIR, "triviaqa/evidence")
        self.file_id_map = file_id_map

    def get_vocab(self):
        with open(join(self.directory, "vocab.txt"), "r", encoding='utf8') as f:
            return {x.strip() for x in f}

    def load_word_vectors(self, vec_name):
        filename = join(self.directory, vec_name + "_pruned.pkl")
        if exists(filename):
            with open(filename, "rb"):
                return pickle.load(filename)
        else:
            return None

    def list_documents(self):
        if self.file_id_map is not None:
            return list(self.file_id_map.keys())
        f = []
        for dirpath, dirnames, filenames in walk(self.directory):
            if dirpath == self.directory:
                # Exclude files in the top level dir, like the vocab file
                continue
            if self.directory != dirpath:
                rel_path = relpath(dirpath, self.directory)
                f += [join(rel_path, x[:-4]) for x in filenames]
            else:
                f += [x[:-4] for x in filenames]
        return f

    def get_document_from_file_path(self, file_id, n_tokens=None, flat=False):
        with open(file_id, "r", encoding='utf8') as f:
            if n_tokens is None:
                text = f.read()
                if flat:
                    return [x for x in self._split_all.split(text) if len(x) > 0]
                else:
                    paragraphs = []
                    for para in self._split_para.split(text):
                        paragraphs.append([sent.split(" ") for sent in para.split("\n")])
                    return paragraphs
            else:
                paragraphs = []
                paragraph = []
                cur_tokens = 0
                for line in f:
                    if line == "\n":
                        if not flat and len(paragraph) > 0:
                            paragraphs.append(paragraph)
                            paragraph = []
                    else:
                        sent = line.split(" ")
                        sent[-1] = sent[-1].rstrip()
                        if len(sent) + cur_tokens > n_tokens:
                            if n_tokens != cur_tokens:
                                paragraph.append(sent[:n_tokens-cur_tokens])
                            break
                        else:
                            paragraph.append(sent)
                            cur_tokens += len(sent)
                if flat:
                    return flatten_iterable(paragraph)
                else:
                    if len(paragraph) > 0:
                        paragraphs.append(paragraph)
                    return paragraphs

    def get_document(self, doc_id, n_tokens=None, flat=False):
        if self.file_id_map is None:
            file_id = doc_id
        else:
            file_id = self.file_id_map.get(doc_id)

        if file_id is None:
            return None

        file_id = join(self.directory, file_id + ".txt")
        if not exists(file_id):
            return None

        return self.get_document_from_file_path(file_id, n_tokens=n_tokens, flat=flat)


evidence_corpus.TriviaQaEvidenceCorpusTxt = PatchedEvidenceCorpusTxt