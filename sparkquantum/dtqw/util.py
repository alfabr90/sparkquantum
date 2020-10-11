# def coin_state(a, b):
#     """Build a coin state.

#     Parameters''
#     ----------
#     a : int, float or complex
#         First amplitude of the coin state.
#     b : int, float or complex
#         Second amplitude of the coin state.

#     Returns
#     -------
#     array-like
#         The coin state.

#     """
#     ca = complex(a)
#     cb = complex(b)

#     if round(abs(ca) ** 2 + abs(cb) ** 2) != 1:
#         raise ValueError("")

#     return [ca, cb]


# def bell_state(x, y):
#     """Build a Bell state considering the following formula:

#         ``|B(x,y)> = (|0,y> + (-1)^x |0,¬y>) / sqrt(2)``

#     Parameters
#     ----------
#     x : int or bool
#         Control the signal (plus or minus).
#     y : int or bool
#         Control the state of the second ket.

#     Returns
#     -------
#     array-like
#         The Bell state.

#     """
#     if y:
#         return (0, 1, (-1) ** int(x), 0)
#     else:
#         return (1, 0, 0, (-1) ** int(x))


# def ghz_state(m):
#     """Build a Greenberger-Horne-Zeilinger (GHZ) state considering the following formula:

#         ``|GHZ(m)> = (|0>(x)^m + |1>(x)^m) / sqrt(2)``

#     Parameters
#     ----------
#     m : int
#         The number of quantum subsystems. Must be greater than 2.

#     Returns
#     -------
#     array-like
#         The GHZ state.

#     """
#     if m <= 2:
#         raise ValueError("`m` must be greater than 2")

#     state = [0 for i in range(2 ** m)]

#     state[0] = 1
#     state[len(state) - 1] = 1

#     return tuple(state)
