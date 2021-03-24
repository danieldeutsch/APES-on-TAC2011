# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import re, argparse, spacy, copy
from tqdm import tqdm
from io import open
import json

entity_names = ['PERSON', 'ORG', 'NORP', 'GPE', 'FACILITY', 'FAC']


def create_entities(args, input_summaries):
    nlp = spacy.load('en')

    entities = {}
    missing_entities = {}
    for prefix in tqdm(input_summaries.keys(), total=len(input_summaries), desc='Creating entities'):
        man_made_files = input_summaries[prefix]['man']
        for man_made_file in man_made_files:
            text = input_summaries[prefix]['man'][man_made_file]
            text = clean_text(text)
            doc = nlp(text)
            count = 0
            for ent in doc.ents:
                if ent.label_ in entity_names and ent.end_char - ent.start_char > 1 and len(ent.text.strip()) >= 0:
                    if prefix not in entities:
                        entities[prefix] = {}
                    if man_made_file not in entities[prefix]:
                        entities[prefix][man_made_file] = []
                    if ent.text not in entities[prefix][man_made_file]:
                        entities[prefix][man_made_file].append(ent.text)
                        count += 1

            if count == 0:
                if prefix not in missing_entities:
                    missing_entities[prefix] = []
                missing_entities[prefix].append(man_made_file)

    return entities, missing_entities


def merge_keywords(args, input_summaries, entities):
    nlp = spacy.load('en')
    keywords = {}

    for prefix in tqdm(input_summaries.keys(), total=len(input_summaries), desc='Merging keywords'):
        for man_made_file in input_summaries[prefix]['man']:
            if prefix not in entities or man_made_file not in entities[prefix]: continue
            keywords_nominees = entities[prefix][man_made_file]

            if prefix not in keywords:
                keywords[prefix] = {}
            keywords[prefix][man_made_file] = merge(nlp, keywords_nominees)

    return keywords


class Node:

    def __init__(self, entity, lemma_entity, value, parent):
        self.entity = entity
        self.lemma_entity = lemma_entity
        self.value = value
        self.parent = parent

    def print_me(self):
        print(self.entity, self.lemma_entity, self.value, self.parent)


def merge(nlp, keywords_nominees, graph=None):
    if graph == None: graph = []
    sorted_keywords = sorted(set([i.lower().strip('()') for i in keywords_nominees]), key=lambda x: (len(x), x.lower()),
                             reverse=True)
    sorted_keywords_lemmas = [nlp(key)[:].lemma_ for key in sorted_keywords]
    i = 0
    while i < len(sorted_keywords):
        if sorted_keywords[i] in [x.entity for x in graph]: i += 1; continue
        graph_indexing = len(graph)
        graph.append(Node(sorted_keywords[i], sorted_keywords_lemmas[i], len(graph), None))
        j = 0
        while j < len(graph):
            if graph_indexing != j:
                if (graph[j].lemma_entity == graph[graph_indexing].lemma_entity or re.search(
                        r'\b' + re.escape(graph[graph_indexing].entity) + r'\b', graph[j].entity,
                        flags=re.IGNORECASE) != None or re.search(r'\b' + re.escape(graph[j].entity) + r'\b',
                                                                  graph[graph_indexing].entity,
                                                                  flags=re.IGNORECASE) != None):
                    graph[graph_indexing].parent = j
                    break
            j += 1
        i += 1

    return graph


def graph_to_dict(graph):
    dic = {}
    for x in graph:
        if x.parent == None:
            dic[x.entity] = x.value
        else:
            curr = graph[x.parent]
            while curr.parent != None:
                curr = graph[curr.parent]
            dic[x.entity] = curr.value

    return dic


def create_questions(args, input_summaries, keywords):
    nlp = spacy.load('en')

    questions = {}
    missing_questions = {}
    i = 0
    for prefix in tqdm(input_summaries.keys(), total=len(input_summaries), desc='Creating questions'):
        if prefix not in keywords:
            # Missing entities
            continue

        for man_made_file in keywords[prefix]:
            text = input_summaries[prefix]['man'][man_made_file]
            text = clean_text(text)
            doc = nlp(text)
            sentences = [sent for sent in doc.sents]
            num_questions = 0

            for sent in sentences:
                entities_asked_about_this_sent = []
                curr_keywords = graph_to_dict(keywords[prefix][man_made_file])
                for keyword, keyword_key in curr_keywords.items():
                    if keyword_key in entities_asked_about_this_sent: continue

                    answer = '@entity' + str(keyword_key)
                    question = entitize(sent, curr_keywords, keyword)
                    if '@placeholder' not in question: continue
                    if prefix not in questions:
                        questions[prefix] = {}
                    if man_made_file not in questions[prefix]:
                        questions[prefix][man_made_file] = []
                    question_splitted_by_spaces = ' '.join([w.text for w in nlp(question)])
                    questions[prefix][man_made_file].append({'question': question_splitted_by_spaces, 'answer': answer})
                    entities_asked_about_this_sent.append(keyword_key)
                    num_questions += 1

            if num_questions == 0:
                if prefix not in missing_questions:
                    missing_questions[prefix] = []
                missing_questions[prefix].append(man_made_file)

    return questions, missing_questions


def create_summaries_with_keywords(args, input_summaries, keywords):
    summaries_with_keywords = {}

    nlp = spacy.load('en')

    for prefix in tqdm(input_summaries.keys(), total=len(input_summaries), desc='Creating summaries with keywords'):
        if prefix not in keywords:
            # Missing entities
            continue

        for questioning_doc in keywords[prefix]:
            for answering_doc in input_summaries[prefix]['machine']:
                text = input_summaries[prefix]['machine'][answering_doc]
                text = clean_text(text)
                doc = nlp(text)
                curr_keywords = keywords[prefix][questioning_doc]
                answering_doc_keywords = []

                for ent in doc.ents:
                    if ent.label_ in entity_names and ent.end_char - ent.start_char > 1:
                        answering_doc_keywords.append(ent.lemma_)
                merged_keywords = graph_to_dict(merge(nlp, answering_doc_keywords, copy.deepcopy(curr_keywords)))
                text = entitize(doc, merged_keywords)
                summaries_with_keywords[(answering_doc, questioning_doc)] = text

    return summaries_with_keywords


def entitize(doc, keywords, answer=None):
    keywords_as_list = sorted(keywords.keys(), key=lambda x: len(x), reverse=True)
    text = lemmatize_questions_by_keywords(doc, keywords)

    for ent in keywords_as_list:
        ent_token = '@entity' + str(keywords[ent])
        text = re.sub(r'\b' + re.escape(ent) + r'\b', ent_token, text, flags=re.IGNORECASE)
    if answer is not None:
        text = re.sub(r'\b' + re.escape('entity' + str(keywords[answer])) + r'\b', 'placeholder', text,
                      flags=re.IGNORECASE)

    return text


def lemmatize_questions_by_keywords(doc, keywords):
    text_as_list = [t.text for t in doc]
    for ent in keywords:
        for i, w in enumerate(doc):
            if w.lemma_ == ent:
                text_as_list[i] = ent
    text = ' '.join(text_as_list)
    return text


def clean_text(text):
    text = text.replace('.\n', '. ')
    text = text.replace('\n', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace("’", "'")
    text = re.sub("\'s", "", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub("s\'", "", text)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\W|^)([0-9]+)[kK](\W|$)", r"\1\g<2>000\3", text) # better regex provided by @armamut
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu\.s\.a", "USA", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu\.s\.", "USA", text, flags=re.IGNORECASE)
    text = re.sub(r"\busa\b", "USA", text, flags=re.IGNORECASE)
    text = re.sub(r"\bus\b", "USA", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " USA ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " USA ", text, flags=re.IGNORECASE)
    text = re.sub("Los angeles", "LA", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    return text


def write_qa_input(args, input_summaries, keywords, questions, summaries_with_keywords, output_file):
    with open(output_file, 'w') as out:
        for prefix in tqdm(input_summaries.keys(), total=len(input_summaries)):
            if prefix not in questions:
                # Missing entities
                continue

            for questioning_doc in questions[prefix].keys():
                curr_questions = questions[prefix][questioning_doc]
                doc_questions, doc_answers = zip(*[[qa['question'], qa['answer']] for qa in curr_questions])
                for answering_doc in input_summaries[prefix]['machine']:
                    if answering_doc == questioning_doc: continue
                    text = summaries_with_keywords[(answering_doc, questioning_doc)]
                    cands = ["@entity" + str(i) for i in graph_to_dict(keywords[prefix][questioning_doc]).values()]

                    out.write(json.dumps({
                        'answering_doc': answering_doc,
                        'questioning_doc': questioning_doc,
                        'cands': cands,
                        'doc_questions': doc_questions,
                        'doc_answers': doc_answers,
                        'text': text
                    }) + '\n')


def main():
    parser = argparse.ArgumentParser(description='justifying APES on TAC2011')
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--metadata-file', required=True)
    args = parser.parse_args()

    input_summaries = json.load(open(args.input_file, 'r', encoding='cp1252'))
    entities, missing_entities = create_entities(args, input_summaries)
    keywords = merge_keywords(args, input_summaries, entities)
    questions, missing_questions = create_questions(args, input_summaries, keywords)
    summaries_with_keywords = create_summaries_with_keywords(args, input_summaries, keywords)
    write_qa_input(args, input_summaries, keywords, questions, summaries_with_keywords, args.output_file)

    with open(args.metadata_file, 'w') as out:
        out.write(unicode(json.dumps({
            'missing_entities': missing_entities,
            'missing_questions': missing_questions
        }, indent=2), 'utf-8'))


if __name__ == "__main__":
    main()
