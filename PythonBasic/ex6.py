class ex6 : 
    def getString(self) :
        self.itsStr = input("Insert a string\t")

    def printStr(self) :
        print("you type {}".format(self.itsStr))

myObj = ex6()
myObj.getString()
myObj.printStr()