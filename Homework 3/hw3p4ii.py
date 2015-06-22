import hw3
import sys

def main(args):
	guess=hw3.classify_new_email(args[1], int(args[2]))[1]
	print guess

if __name__ == '__main__':
    main(sys.argv)