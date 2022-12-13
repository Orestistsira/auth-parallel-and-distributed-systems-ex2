import math


# Function to find the partition position
def partition(array, other_array, low, high):
    # choose the rightmost element as pivot
    pivot = array[high]

    # pointer for greater element
    i = low - 1

    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # Swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])
            (other_array[i], other_array[j]) = (other_array[j], other_array[i])

    # Swap the pivot element with the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])
    (other_array[i + 1], other_array[high]) = (other_array[high], other_array[i + 1])

    # Return the position from where partition is done
    return i + 1


def quick_sort(array, other_array, low, high):
    if low < high:
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, other_array, low, high)

        # Recursive call on the left of pivot
        quick_sort(array, other_array, low, pi - 1)

        # Recursive call on the right of pivot
        quick_sort(array, other_array, pi + 1, high)


def knn(X, Y, m, n, d, k):

    nidx = [[-1.0 for i in range(k)] for j in range(m)]
    ndist = [[-1.0 for i in range(k)] for j in range(m)]

    distances = [[-1.0 for i in range(m)] for j in range(n)]
    ids = [-1.0 for i in range(n)]

    for y in range(0, n):
        ids[y] = y

    print(ids)

    for x in range(0, m):
        for y in range(0, n):
            summary = 0
            for i in range(0, d):
                summary += math.pow((X[x][i] - Y[y][i]), 2)

            distances[x][y] = math.sqrt(summary)

    print(distances)

    for x in range(0, m):
        xdist = distances[x]
        # pass by value (copy array)
        xid = ids[:]

        quick_sort(xdist, xid, 0, n - 1)

        print(xdist)
        print(xid)

        ndist[x] = xdist[0:k]
        nidx[x] = xid[0:k]

    print(ndist)
    print(nidx)


X = [[1, 2, 3],
     [4, 5, 6]]

# Y = [[1, 5, 2],
#      [3, 2, 4]]

Y = [[3, 2, 4],
     [1, 5, 2]]

knn(X, Y, 2, 2, 3, 1)
