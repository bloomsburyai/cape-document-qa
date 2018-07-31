import numpy as np
from cape_document_qa.cape_docqa_machine_reader import get_production_model_config,CapeDocQAMachineReaderModel

def test_machine_reader():
    import time
    conf = get_production_model_config()
    machine_reader = CapeDocQAMachineReaderModel(conf)

    context = '''"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."'''
    question = "Which NFL team represented the AFC at Super Bowl 50?"

    num_trials = 10
    t1 = time.time()
    for i in range(num_trials):
        print('Iteration {0}'.format(i))
        doc_embedding = machine_reader.get_document_embedding(context)
        start_logits, end_logits = machine_reader.get_logits(question, doc_embedding)
        toks, offs = machine_reader.tokenize(context)
        st, en = offs[np.argmax(start_logits)], offs[np.argmax(end_logits)]
        print(context[st[0]: en[1]])

    t2 = time.time()
    print('full reading takes {} seconds'.format((t2 - t1) / num_trials))

    for i in range(num_trials):
        print('Iteration {0}'.format(i))
        start_logits, end_logits = machine_reader.get_logits(question, doc_embedding)
        toks, offs = machine_reader.tokenize(context)
        st, en = offs[np.argmax(start_logits)], offs[np.argmax(end_logits)]
        print(context[st[0]: en[1]])
    t3 = time.time()
    print('Cached reading takes {} seconds'.format((t3 - t2) / num_trials))
