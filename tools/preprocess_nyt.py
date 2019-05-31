"""
Script from Paulus et al. (2017), for preprocessing NYT dataset.
Supplied by Romain Paulus. Code under his Copyright.
"""
import os
import sys
import tarfile
import re
import multiprocessing
import json
import logging
from glob import glob
from contextlib import closing
import xml.etree.ElementTree as ET
import requests


class Dataset(object):
    """
    Generic dataset object that encapsulates a list of instances.

    The dataset is analogous to a simplified table, whereby each cell can contain arbitrary data types.
    Each row in the table is defined by a tuple.
    The columns in the table are defined by `self.fields`.

    The dataset object supports indexing, iterating, slicing (eg. for iterating over batches), shuffling, and
    serialization/deserialization to/from JSONL.

    Examples:
        >>> Dataset(['Name', 'SSN']).add_example('A', 1)
        Dataset(Name, SSN)
        >>> Dataset(['Name', 'SSN']).add_examples([('A', 1), ('B', 2), ('C', 3)]).data
        [('A', 1), ('B', 2), ('C', 3)]
        >>> Dataset(['Name', 'SSN']).add_examples([('A', 1), ('B', 2), ('C', 3)])[1]
        ('B', 2)
        >>> Dataset(['Name', 'SSN']).add_examples([('A', 1), ('B', 2), ('C', 3)])[1:]
        [('B', 2), ('C', 3)]
        >>> [e for e in Dataset(['Name', 'SSN']).add_examples([('A', 1), ('B', 2), ('C', 3)])]
        [('A', 1), ('B', 2), ('C', 3)]

    """

    def __init__(self, fields):
        """

        Args:

            fields: A tuple of fields in the dataset.
        """
        assert isinstance(fields, tuple) or isinstance(fields, list)
        self.fields = tuple(fields)
        self.data = []

    def add_example(self, *args, **kwargs):
        """
        Adds a single example to the dataset.

        Args:
            *args: tuple arguments for the example, according to dataset order as indicated by `fields`.
            **kwargs: keyword arguments for the example, according to dataset fields as indicated by `fields`.

        Returns: the added example as a `tuple`

        Examples:
            >>> Dataset(['name', 'ssn']).add_example('Adam', 123).data
            [('Adam', 123)]
            >>> Dataset(['name', 'ssn']).add_example(name='Adam', ssn=123).data
            [('Adam', 123)]

        """
        if not kwargs:
            tup = args
        else:
            tup = {}
            for i, f in enumerate(self.fields):
                if i < len(args):
                    tup[f] = args[i]
                if f in kwargs:
                    tup[f] = kwargs[f]
            tup = tuple(tup[f] for f in self.fields)
        assert len(tup) == len(
            self.fields
        ), "Expected {} fields, example only contains {}".format(
            len(self.fields), len(tup)
        )
        self.data.append(tup)
        return self

    def add_examples(self, rows):
        """
        Adds many examples to the dataset

        Args:
            rows: List of tuples to add to the dataset. Each tuple should be in `field` order.

        Returns: the modified dataset

        Examples:
            >>> Dataset(['Name', 'SSN']).add_examples([('A', 1), ('B', 2), ('C', 3)]).data
            [('A', 1), ('B', 2), ('C', 3)]

        """
        assert isinstance(rows, list)
        for r in rows:
            assert isinstance(r, tuple)
            self.add_example(*r)
        return self

    def __len__(self):
        """

        Returns: number of examples in the dataset

        """
        return len(self.data)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(self.fields))

    def __getitem__(self, item):
        """

        Args:
            item: An integer index or a slice (eg. 2, 1:, 1:5)

        Returns: tuple(s) corresponding to the instance(s) at index/indices `item`.

        """
        return self.data[item]

    def __setitem__(self, item, tup):
        """

        Args:
            item: An integer index or a slice (eg. 2, 1:, 1:5)
            tup: tuple arguments for the example, according to dataset order as indicated by `fields`.

        """
        self.data[item] = tup

    def __iter__(self):
        """

        Returns: A iterator over the instances in the dataset

        """
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def construct(cls, **kwargs):
        """

        Generic dataset loading method. This method must be implemented by
        user datasets.

        Args:
            **kwargs: key word arguments for the construct method

        Returns: instance of a Dataset from a custom format.

        """
        raise NotImplementedError()

    def split(self, *proportions):
        """

        Splits a dataset based on proportions.

        """
        assert abs(sum(proportions) - 1) < 1e-9
        start = 0
        splits = []
        for p in proportions:
            assert isinstance(p, float)
            assert p < 1
            size = int(len(self) * p)
            end = start + size
            ex = self[start:end]
            splits.append(self.__class__(self.fields).add_examples(ex))
            start = end
        # add the remaining to the last split
        splits[-1].add_examples(self[start:])
        return splits

    @classmethod
    def deserialize(cls, fname):
        """
        Deserializes a Dataset object from a JSONL file.

        Args:
            fname: The JSONL formatted file from which to load the dataset

        Returns: loaded Dataset instance

        """
        assert isinstance(fname, str)
        with open(fname) as f:
            header = next(f).rstrip("\n")
            fields = json.loads(header)
            d = cls(fields)
            for line in f:
                row = json.loads(line.rstrip("\n"))
                d.add_example(*row)
            return d

    def serialize(self, fname):
        """
        Serializes a Dataset object to a JSONL file

        Args:
            fname: The JSONL formatted file to write the dataset to

        """
        assert isinstance(fname, str)
        with open(fname, "w") as f:
            f.write(json.dumps(self.fields) + "\n")
            for example in self:
                f.write(json.dumps(example) + "\n")


class NYTSummarization(Dataset):

    FIELD_NAMES = ["article_tokens", "summary_tokens", "pointers"]

    # Years before 1996 don't have any abstract/summaries, so we can skip them
    YEARS_WITH_ABSTRACTS = list(range(1996, 2007 + 1))

    # train/dev/test splits
    SPLITS = [0.9, 0.05, 0.05]

    @classmethod
    def construct(
        cls, location, fname="nyt-corpus.tgz", multiprocess=True, n_pool_workers=None
    ):
        """Create the NYT summarization dataset.

        :param multiprocess: determines whether multiprocessing is used for
            parsing documents in parallel.

        :param n_pool_workers: For multiprocess only. Determines how many
            processes are used to perform parallel operations.
            default: CPU_COUNT * 2
        """

        # Set logging
        logger = logging.getLogger("nyt")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(ch)

        dataset = cls(cls.FIELD_NAMES)

        # Set default value
        if multiprocess and n_pool_workers is None:
            n_pool_workers = multiprocessing.cpu_count() * 2

        # Extract members of main tar file
        logger.info("Extract full tar...")
        print(fname)
        with tarfile.open(fname, "r:gz") as tar:
            for year in cls.YEARS_WITH_ABSTRACTS:
                for month in range(1, 12 + 1):
                    tar_path = os.path.join(
                        "nyt_corpus", "data", str(year), "%02d.tgz" % month
                    )
                    dest_path = os.path.join(location, os.path.splitext(tar_path)[0])
                    if not os.path.isdir(dest_path):
                        try:
                            logger.info("Extract %s..." % tar_path)
                            tar.extract(tar_path, path=location)
                        except KeyError:  # No archive for this month/year
                            pass
        logger.info("Done extracting")

        # Extract 2nd level tar files, possibly in parallel
        iter_tar_files = glob(
            os.path.join(location, "nyt_corpus", "data", "*", "*.tgz")
        )

        # Filter existing paths
        logger.info("Filter tars...")
        iter_tar_files = [
            path
            for path in iter_tar_files
            if not os.path.isdir(os.path.splitext(path)[0])
        ]
        logger.info("Total tars: %s" % len(iter_tar_files))

        logger.info("Extract individual tars... (this might take a while)")
        if iter_tar_files:
            if multiprocess:
                with closing(multiprocessing.Pool(n_pool_workers)) as pool:
                    pool.map(_extract_tar, iter_tar_files)
            else:
                for path in iter_tar_files:
                    _extract_tar(path)
        logger.info("Done extracting")

        logger.info("List files...")
        # List all individual XML files
        iter_docs = glob(
            os.path.join(location, "nyt_corpus", "data", "*", "*", "*", "*.xml")
        )
        #  iter_docs = iter_docs[:100]
        logger.info("Parse documents (this might take a while)...")
        if multiprocess:
            with closing(multiprocessing.Pool(n_pool_workers)) as pool:
                fields_list = pool.map(_parse_nyt_document, iter_docs)
        else:
            fields_list = [_parse_nyt_document(path) for path in iter_docs]
        logger.info("Done parsing.")

        logger.info("Add examples...")
        for fields in fields_list:
            if fields is not None:
                dataset.add_example(*fields)
        logger.info("Done adding")

        return dataset.split(*cls.SPLITS)


def _extract_tar(path):
    """Extract a tar file of the NYT dataset."""

    extraction_path = os.path.dirname(path)
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(extraction_path)

    os.remove(path)  # No need to keep another copy of it


def _parse_nyt_document(path, clean_summary=True, remove_corrections=True):
    """Parse an XML document and return relevant fields."""
    try:
        xml = ET.parse(path)
    except:
        print("Found empty xml, skipping")
        return None
    headline_el = xml.find(".//hedline")
    byline_el = xml.find(".//byline[@class='print_byline']")
    article_el = xml.find(".//block[@class='full_text']")
    summary_el = xml.find(".//abstract")

    if any(el is None for el in [article_el, summary_el, headline_el]):
        return None

    headline = " ".join(
        [text.strip() for text in headline_el.itertext() if text.strip()]
    )
    summary = " ".join([text.strip() for text in summary_el.itertext() if text.strip()])
    article = "\n".join(
        [text.strip() for text in article_el.itertext() if text.strip()]
    )
    if byline_el is not None:
        byline = " ".join(
            [text.strip() for text in byline_el.itertext() if text.strip()]
        )
    else:
        byline = ""

    # Remove useless bits at the end of the summary
    if clean_summary:
        summary = re.sub(r" ?\(.\)", " ", summary)
        summary = re.sub(
            r"[;:] (photo|graph|chart|map|table|drawing|listing|interview)s?(?=([;:]| *$))",
            " ",
            summary,
        )
        summary = summary.strip()

    # TODO: add entities too
    # TODO: convert numbers to 0!!!!!!
    headline = corenlp_tokenize(headline)

    # Headline is sometimes duplicate
    if headline[len(headline) // 2 :] == headline[: len(headline) // 2]:
        headline = headline[len(headline) // 2 :]

    byline = corenlp_tokenize(byline)
    article = corenlp_tokenize(article)
    summary, entities = corenlp_tokenize(summary, with_entities=True)

    concat = headline + ["***END_HEADLINE***"] + byline + ["***END_BYLINE***"] + article

    pointers = get_nyt_pointers(concat, summary, entities)

    return concat, summary, pointers


def corenlp_tokenize(
    text,
    corenlp_server_url="http://localhost:9000",
    with_entities=False,
    convert_numbers=True,
):
    CORENLP_MAX_LEN = 100000
    ENTITIES_TYPE = {"PERSON", "LOCATION", "ORGANIZATION", "MISC"}

    annotators = ["tokenize", "ssplit"]

    if with_entities:
        entities = set()
        annotators.append("ner")

    response = requests.post(
        corenlp_server_url
        + '?properties={"annotators":"'
        + ",".join(annotators)
        + '","outputFormat":"json"}',
        data=text[:CORENLP_MAX_LEN].encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded ; charset=UTF-8"},
    )
    assert response.ok, text

    tokens = []
    for sent in response.json()["sentences"]:
        tokens += [t["word"] for t in sent["tokens"]]
        if with_entities:
            entities.update(
                [t["word"].lower() for t in sent["tokens"] if t["ner"] in ENTITIES_TYPE]
            )

    if convert_numbers:
        for idx, _ in enumerate(tokens):
            tokens[idx] = re.sub(r"\d", "0", tokens[idx])

    if with_entities:
        return tokens, entities
    else:
        return tokens


def get_nyt_pointers(article, summary, entities):

    entities_idx = {}

    for idx, token in enumerate(article):
        if token.lower() in entities:
            entities_idx[token.lower()] = idx

    return [entities_idx.get(token.lower(), -1) for token in summary]


if __name__ == "__main__":
    train, val, test = NYTSummarization.construct(".")
    train.serialize("train")
    val.serialize("val")
    test.serialize("test")
