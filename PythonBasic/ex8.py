def checkLowerCase(str) :
    for i in range(0, len(str)) :
        if str[i] >= 'a' and str[i] <= 'z' : return True
    return False

def checkUpperCase(str) :
    for i in range(0, len(str)) :
        if str[i] >= 'A' and str[i] <= 'Z' : return True
    return False

def checkNumericCase(str):
    for i in range(0, len(str)) :
        if str[i] >= '0' and str[i] <= '9' : return True
    return False

def checkSpecialCase(str):  
    for i in range(0, len(str)) :
        if str[i] == '$' or str[i] == '#' or str[i] == '@' : return True
    return False

def checkLength(str) :
    length = len(str)
    return length >= 6 and length <= 12

def checkPassword(passwd) :
    if not checkLowerCase(passwd) : print("no character in a - z")
    elif not checkUpperCase(passwd) : print("no character in A - Z")
    elif not checkNumericCase(passwd) : print("no character in 0 - 9")
    elif not checkSpecialCase(passwd) : print("no character in {$, #, @}")
    elif not checkLength(passwd) : print("password length not match")
    else : print("Password satisfied")

    # return True

passwd = input("Insert a password:\t")
checkPassword(passwd)

