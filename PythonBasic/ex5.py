def convertNum(num) : 
    outNum = 0
    while num > 0 :
        outNum = outNum * 10 + num % 10
        num = num / 10
    return outNum

def isPrime(num) :
    for i in range(2, num / 2 + 1) :
        if num % i == 0 : return False
    return True

def main() :
    num = int(input("insert a number\t"))
    if isPrime(num) and num == convertNum(num) :
        print("{} is a Palindrome number", num)
    else :  print("{} is NOT a Palindrome number".format(num))

main()