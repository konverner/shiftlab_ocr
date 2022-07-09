import os

class Font:
    def __init__(self, path, cyr_small, cyr_capit, digits, chars=[], size_coef=1):
        self.path = path
        self.chars = chars
        self.size_coef = size_coef

        if cyr_small:
            for char in range(1072, 1104):
                self.chars.append(chr(char))
            self.chars.append('ё')
        if cyr_capit:
            for char in range(1040, 1072):
                self.chars.append(chr(char))
        if digits:
            for i in range(0, 10):
                self.chars.append(str(i))

    def isValid(self, string):
        chars = set(string)
        for char in string:
            if char not in self.chars:
                return False
        return True

    def __str__(self):
        result = self.path + '\n'
        for char in self.chars:
            result += char + ' '
        return result

dirname = os.path.dirname(__file__)
DIR = os.path.join(dirname, 'content')
f = [Font(os.path.join(DIR,'Lemon Tuesday.otf'),True,True,False,list('.,;:"()? '),0.8), \
Font(os.path.join(DIR,'ofont.ru_BetinaScriptCTT.ttf'),True,True,True,list('+.,;:"-%$[]()«»!?/ '),0.6), \
Font(os.path.join(DIR,'ofont.ru_Denistina.ttf'),True,True,True,list('+.,;-:"()«/»!? '),0.8), \
Font(os.path.join(DIR,'ofont.ru_Eskal.ttf'),True,True,True,list('+.,;-:"()/?! '),1), \
Font(os.path.join(DIR,'ofont.ru_Rozovii Chulok.ttf'),True,True,True,list('+.,;-:"()«/»!? '),0.8), \
Font(os.path.join(DIR,'ofont.ru_Shlapak Script.otf'),True,True,True,list('+.,;-:"()/!? Regular'),1), \
Font(os.path.join(DIR,'werner4.ttf'),True,True,False,list('+.,;:"()!? '),1), \
Font(os.path.join(DIR,'werner6.ttf'),True,True,False,list('.,;:"()!? '),0.9), \
Font(os.path.join(DIR,'werner7.ttf'),True,True,False,list('.,;:"()!? '),1), \
Font(os.path.join(DIR,'werner13.ttf'),True,True,False,list('.,;:"()«»! '),1), \
Font(os.path.join(DIR,'werner14.ttf'),True,False,True,list(' %,./[]:;'),1), \
Font(os.path.join(DIR,'Jayadhira.ttf'),False,False,True,list(' +%,.-/()[]:;'),0.5), \
Font(os.path.join(DIR,'werner11.ttf'),True,True,False,list(' %,./[]:;'),1), \
Font(os.path.join(DIR,'werner15.ttf'),True,False,True,list('+.,;:"()! '),1), \
Font(os.path.join(DIR,'werner16.ttf'),True,False,True,list(' +%,./()[]:;'),1), \
Font(os.path.join(DIR,'werner17.ttf'),True,True,False,list(' +%,./()[]:;'),1), \
Font(os.path.join(DIR,'bimbo.regular.ttf'),True,True,False,list(' +%,.-/()[]:;'),0.8), \
Font(os.path.join(DIR,'amandasignature.ttf'),False,False,True,list(' +%,.-/()[]:;'),0.5), \
Font(os.path.join(DIR,'mathilde.regular.otf'),False,False,True,list(' +%,.-/()[]:;'),0.5), \
Font(os.path.join(DIR,'werner2.ttf'),True,True,False,list(' ?!,.:;"()+[]'),1.4), \
Font(os.path.join(DIR,'werner3.ttf'),True,True,False,list(' ?!,.:;"()+[]'),1.4), \
Font(os.path.join(DIR,'werner5.ttf'),False,True,False,list(' ?!,.:;/-+[]'),1.4), \
Font(os.path.join(DIR,'werner20.ttf'),True,False,True,list(' %,./()[]:;'),1.4), \
Font(os.path.join(DIR,'werner21.ttf'),True,False,True,list(' %,./()[]:;'),1.4), \
Font(os.path.join(DIR,'werner22.ttf'),True,False,True,list('!%/,./()[]:;'),1.4), \
Font(os.path.join(DIR,'werner23.ttf'),True,False,True,list('!%/,./()[]:;'),1.4), \
Font(os.path.join(DIR,'werner31.ttf'),True,False,True,list('!%+,./"[]:;'),1.4), \
Font(os.path.join(DIR,'werner36.ttf'),True,False,False,list('!%+,./"()[]:;'),1.4), \
Font(os.path.join(DIR,'werner37.ttf'),True,False,True,list('!%+,./"()[]:;'),1.4), \
Font(os.path.join(DIR,'werner10.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'werner30.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'werner39.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'werner40.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'werner41.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'werner42.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'werner43.ttf'),True,True,False,list(''),1.4), \
Font(os.path.join(DIR,'ofont.ru_Marutya.ttf'),True,True,False,list(''),0.7) ]