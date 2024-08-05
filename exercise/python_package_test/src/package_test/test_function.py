import random

def add_random_number(input_n=0):
    random_n = random.randint(0,9)
    print("input:{} random:{}".format(input_n,random_n))
    return input_n+random_n