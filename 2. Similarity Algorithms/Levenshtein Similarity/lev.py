import numpy


def minimum(a,b,c):
    if a<=b and a<=c:
        return a
    if b<=c and b<=a:
        return b
    if c<=a and c<=b:
        return c

def max(a,b):
    if(a < b):
        return b
    else:
        return a

def normalize(X,size):

    if len(X) < size:
        fark = size - len(X)
        for i in range(fark):
            X = X + " "

    return X


def LevenshteinMesafesi(A,B):
    K = numpy.zeros((len(A) + 1, len(B) + 1))   # AxB boyutlarinda 0 matrisi olustur
    A_len = len(A)
    B_len = len(B)

    for i in range(A_len + 1):
        K[i][0] = i
    for i in range(B_len + 1):
        K[0][i] = i


    silme = 0
    ekleme= 0
    yerdegistirme = 0

    for i in range(1, A_len + 1):
        for j in range (1, B_len + 1):
            if A[i-1] == B[j-1]:
                K[i][j] = K[i-1][j-1]
            else:
                silme =  K[i-1][ j] + 1
                ekleme = K[i][ j-1] + 1
                yerdegistirme = K[i-1][ j-1] + 1
                K[i][j] = minimum(silme,ekleme,yerdegistirme)

    return int(K[A_len][B_len])

if __name__ == "__main__":
    kelime_1 = input("İlk kelime: ")
    kelime_2 = input("İkinci kelime: ")
    max_len = max(len(kelime_1), len(kelime_2))
    mesafe = LevenshteinMesafesi(kelime_1, kelime_2)
    print(f"Levenshtein mesafesi: {mesafe}")
    print(f"Benzerlik oranı: {1 - mesafe / max_len if max_len else 1:.2%}")
