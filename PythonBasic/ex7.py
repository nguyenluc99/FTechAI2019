def main() :
    iniStr = input("insert a string\t")
    # if len(checkValidEmail())
    lst = iniStr.split('@')
    if len(lst) == 2 :
        print("Username : {}, company : {}", lst[0], lst[1])
    else : print("Not valid email")

main()
