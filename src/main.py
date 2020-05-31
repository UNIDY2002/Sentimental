import sys

if len(sys.argv) == 2 and sys.argv[1].lower() in ['mlp', 'cnn', 'lstm', 'bayes']:
    if sys.argv[0].lower() != 'main.py':
        print('Warning: The very first argument is not "main.py", which might lead to unexpected results.')
    __import__(sys.argv[1].lower())
else:
    print('''Usage: python main.py MODE
       where MODE is one of mlp, cnn, lstm and bayes.''')
