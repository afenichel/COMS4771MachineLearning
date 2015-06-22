import hw3
import sys

def main(args):
	word_bag=hw3.bag_of_words(int(args[1]))[0]
	print word_bag

if __name__ == '__main__':
    main(sys.argv)