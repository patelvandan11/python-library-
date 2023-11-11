# NumPy Cheat Sheet

The Numpy library is the core library for scientific computing in python. It provides a high-performance multidimensional array object , and tools for working with array. 


```python
import numpy as np
```

## creating Arrays 


```python
a=np.array([1,2,3])
a
```




    array([1, 2, 3])




```python
b = np.array([(1.5,2,3), (4,5,6)], dtype = float)
b
```




    array([[1.5, 2. , 3. ],
           [4. , 5. , 6. ]])




```python
c=np.array([[(1.5,2,3),(4,5,6)],[(3,2,1),(4,5,6)]],dtype=float)
c
```




    array([[[1.5, 2. , 3. ],
            [4. , 5. , 6. ]],
    
           [[3. , 2. , 1. ],
            [4. , 5. , 6. ]]])



Initial Placeholders


```python
np.zeros((3,4),dtype=int )  # must be add any data type 
#Create an array of zeros
```




    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])




```python
np.ones((2,3,4),dtype=np.int16)   # Create an array of ones
```




    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],
    
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]], dtype=int16)




```python
d=np.arange(18,25,5)  # Create an array of evenly spaced values (step value)
d
```




    array([18, 23])




```python
np.linspace(1,2,10) #Create an array of evenly spaced values (number of samples) : ( num1,num2,n: no. of part)
```




    array([1.        , 1.11111111, 1.22222222, 1.33333333, 1.44444444,
           1.55555556, 1.66666667, 1.77777778, 1.88888889, 2.        ])




```python
e = np.full((2,2),7)
e
```




    array([[7, 7],
           [7, 7]])




```python
f = np.eye(2) 
f
```




    array([[1., 0.],
           [0., 1.]])




```python
np.random.random((3,3,3)) 
```




    array([[[0.77695909, 0.59350866, 0.086928  ],
            [0.25215302, 0.68944534, 0.04360663],
            [0.05980313, 0.45107074, 0.39653335]],
    
           [[0.18754105, 0.85534883, 0.55146437],
            [0.78220368, 0.89808199, 0.25846412],
            [0.4628483 , 0.34213515, 0.02164985]],
    
           [[0.29187601, 0.33512378, 0.59653808],
            [0.5311824 , 0.37966835, 0.70140715],
            [0.04216022, 0.51464813, 0.19853695]]])




```python
np.empty((3,2)) 
```




    array([[1.5, 2. ],
           [3. , 4. ],
           [5. , 6. ]])



# Saving & Loading On Disk


```python
np.save( "my_array", a)

```


```python
np.savez("array.npz", a, b)

```


```python
np.load("my_array.npz")
```




    <numpy.lib.npyio.NpzFile at 0x194c7808fd0>



# Saving & Loading Text Files


```python
np.loadtxt("myfile.txt")

```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    c:\Desktop\vandan\code\py\1Numpy\numpy.ipynb Cell 22 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Desktop/vandan/code/py/1Numpy/numpy.ipynb#X41sZmlsZQ%3D%3D?line=0'>1</a> np.loadtxt("myfile.txt")
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\npyio.py:1356, in loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, max_rows, quotechar, like)
       1353 if isinstance(delimiter, bytes):
       1354     delimiter = delimiter.decode('latin1')
    -> 1356 arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
       1357             converters=converters, skiplines=skiprows, usecols=usecols,
       1358             unpack=unpack, ndmin=ndmin, encoding=encoding,
       1359             max_rows=max_rows, quote=quotechar)
       1361 return arr
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\npyio.py:975, in _read(fname, delimiter, comment, quote, imaginary_unit, usecols, skiplines, max_rows, converters, ndmin, unpack, dtype, encoding)
        973     fname = os.fspath(fname)
        974 if isinstance(fname, str):
    --> 975     fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
        976     if encoding is None:
        977         encoding = getattr(fh, 'encoding', 'latin1')
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\_datasource.py:193, in open(path, mode, destpath, encoding, newline)
        156 """
        157 Open `path` with `mode` and return the file object.
        158 
       (...)
        189 
        190 """
        192 ds = DataSource(destpath)
    --> 193 return ds.open(path, mode, encoding=encoding, newline=newline)
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\_datasource.py:533, in DataSource.open(self, path, mode, encoding, newline)
        530     return _file_openers[ext](found, mode=mode,
        531                               encoding=encoding, newline=newline)
        532 else:
    --> 533     raise FileNotFoundError(f"{path} not found.")
    

    FileNotFoundError: myfile.txt not found.



```python
np.genfromtxt( "my_file.csv", delimiter=',' )
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    c:\Desktop\vandan\code\py\1Numpy\numpy.ipynb Cell 23 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Desktop/vandan/code/py/1Numpy/numpy.ipynb#X42sZmlsZQ%3D%3D?line=0'>1</a> np.genfromtxt( "my_file.csv", delimiter=',' )
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\npyio.py:1977, in genfromtxt(fname, dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values, filling_values, usecols, names, excludelist, deletechars, replace_space, autostrip, case_sensitive, defaultfmt, unpack, usemask, loose, invalid_raise, max_rows, encoding, ndmin, like)
       1975     fname = os_fspath(fname)
       1976 if isinstance(fname, str):
    -> 1977     fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
       1978     fid_ctx = contextlib.closing(fid)
       1979 else:
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\_datasource.py:193, in open(path, mode, destpath, encoding, newline)
        156 """
        157 Open `path` with `mode` and return the file object.
        158 
       (...)
        189 
        190 """
        192 ds = DataSource(destpath)
    --> 193 return ds.open(path, mode, encoding=encoding, newline=newline)
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\_datasource.py:533, in DataSource.open(self, path, mode, encoding, newline)
        530     return _file_openers[ext](found, mode=mode,
        531                               encoding=encoding, newline=newline)
        532 else:
    --> 533     raise FileNotFoundError(f"{path} not found.")
    

    FileNotFoundError: my_file.csv not found.



```python
np.savetxt("myarray.txt",a,delimiter=" ")
```

# Asking For Help


```python
np.info(np.ndarray.dtype)
```

    Data-type of the array's elements.
    
    .. warning::
    
        Setting ``arr.dtype`` is discouraged and may be deprecated in the
        future.  Setting will replace the ``dtype`` without modifying the
        memory (see also `ndarray.view` and `ndarray.astype`).
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d : numpy dtype object
    
    See Also
    --------
    ndarray.astype : Cast the values contained in the array to a new data-type.
    ndarray.view : Create a view of the same data but a different data-type.
    numpy.dtype
    
    Examples
    --------
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <type 'numpy.dtype'>
    

# Inspecting Your Array


```python
a.shape
```




    (3,)




```python
len(a)
```




    3




```python
b.ndim
```




    2




```python
e.size
```




    4




```python
d.dtype
```




    dtype('int32')




```python
d.dtype.name
```




    'int32'




```python
d.astype(int)
```




    array([18, 23])



# data types


```python
np.int64 #Signed 64-bit integer types

```




    numpy.int64




```python
np.float32 #Standard double-precision floating point

```




    numpy.float32




```python
np.Complex        #Complex numbers represented by 128 floats

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    c:\Desktop\vandan\code\py\1Numpy\numpy.ipynb Cell 38 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Desktop/vandan/code/py/1Numpy/numpy.ipynb#X31sZmlsZQ%3D%3D?line=0'>1</a> np.Complex        #Complex numbers represented by 128 floats
    

    File c:\Users\vanda_6or80vl\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\__init__.py:320, in __getattr__(attr)
        317     from .testing import Tester
        318     return Tester
    --> 320 raise AttributeError("module {!r} has no attribute "
        321                      "{!r}".format(__name__, attr))
    

    AttributeError: module 'numpy' has no attribute 'Complex'



```python
# np.bool_(a)
np.bool_()
```




    False




```python
np.object_
```




    numpy.object_




```python
np.string_
```




    numpy.bytes_




```python
np.unicode_
```




    numpy.str_



# Array mathematics

## Arithmetic operations 


```python
g= a-b
g
```




    array([[-0.5,  0. ,  0. ],
           [-3. , -3. , -3. ]])




```python
np.subtract(a,b)
```




    array([[-0.5,  0. ,  0. ],
           [-3. , -3. , -3. ]])




```python
g=a+b
g
```




    array([[2.5, 4. , 6. ],
           [5. , 7. , 9. ]])




```python
np.add(a,b)
```




    array([[2.5, 4. , 6. ],
           [5. , 7. , 9. ]])




```python
g=a/b
g
```




    array([[0.66666667, 1.        , 1.        ],
           [0.25      , 0.4       , 0.5       ]])




```python
np.divide(a,b)
```




    array([[0.66666667, 1.        , 1.        ],
           [0.25      , 0.4       , 0.5       ]])




```python
g=a*b
g
```




    array([[ 1.5,  4. ,  9. ],
           [ 4. , 10. , 18. ]])




```python
np.multiply(a,b)
```




    array([[ 1.5,  4. ,  9. ],
           [ 4. , 10. , 18. ]])




```python
np.exp(b)
```




    array([[  4.48168907,   7.3890561 ,  20.08553692],
           [ 54.59815003, 148.4131591 , 403.42879349]])




```python
np.sqrt(a)
```




    array([1.        , 1.41421356, 1.73205081])




```python
np.sin(a)
```




    array([0.84147098, 0.90929743, 0.14112001])




```python
np.cos(a)
```




    array([ 0.54030231, -0.41614684, -0.9899925 ])




```python
np.log(a)
```




    array([0.        , 0.69314718, 1.09861229])




```python
f=np.eye(2)
f
```




    array([[1., 0.],
           [0., 1.]])




```python
e.dot(f)
```




    array([[7., 7.],
           [7., 7.]])



## comparison


```python
a==b
```




    array([[False,  True,  True],
           [False, False, False]])




```python
a<2
```




    array([ True, False, False])




```python
np.array_equal(a,b)
```




    False



## aggregate functions


```python
a.sum()
```




    6




```python
a.min()
```




    1




```python
b.max()
```




    6.0




```python
a.max(axis=0)
```




    3




```python
b.cumsum(axis=1)
```




    array([[ 1.5,  3.5,  6.5],
           [ 4. ,  9. , 15. ]])




```python
a.mean()
```




    2.0




```python
np.median(a)
```




    2.0




```python
np.corrcoef(a)
```




    1.0




```python
np.std(b)
```




    1.5920810978785667



# Copying Arrays


```python
h=a.view()
h
```




    array([1, 2, 3])




```python
np.copy(a)
```




    array([1, 2, 3])




```python
h=a.copy()
h
```




    array([1, 2, 3])



# sorting array


```python
a.sort()
a
```




    array([1, 2, 3])




```python
b.sort()
b
```




    array([[1.5, 2. , 3. ],
           [4. , 5. , 6. ]])




```python
c.sort(axis=0)
c
```




    array([[[1.5, 2. , 1. ],
            [4. , 5. , 6. ]],
    
           [[3. , 2. , 3. ],
            [4. , 5. , 6. ]]])



# Subsetting ,slicing ,indexing 

## subsetting 


```python
a[2]
```




    3




```python
b[1,2]
```




    6.0



## slicing


```python
a[0:2]
```




    array([1, 2])




```python
b[0:2,1]
```




    array([2., 5.])




```python
b[:1]
```




    array([[1.5, 2. , 3. ]])




```python
c[1,...]
```




    array([[3., 2., 3.],
           [4., 5., 6.]])




```python
a[: :-1]
```




    array([3, 2, 1])



## Boolean indexing


```python
a[a<2]
```




    array([1])



## fancy indexing


```python
b[[1,0,1,0],[0,1,2,0]]
```




    array([4. , 2. , 6. , 1.5])




```python
b[[1,0,1,0,]][:,[0,1,2,0]]
```




    array([[4. , 5. , 6. , 4. ],
           [1.5, 2. , 3. , 1.5],
           [4. , 5. , 6. , 4. ],
           [1.5, 2. , 3. , 1.5]])



# Array manipulation

## Transposing array


```python
i=np.transpose(b)
i
```




    array([[1.5, 4. ],
           [2. , 5. ],
           [3. , 6. ]])




```python
b
```




    array([[1.5, 2. , 3. ],
           [4. , 5. , 6. ]])




```python
i.T
```




    array([[1.5, 2. , 3. ],
           [4. , 5. , 6. ]])



## changing Array shape


```python
b.ravel()
```




    array([1.5, 2. , 3. , 4. , 5. , 6. ])




```python
g.reshape(3,-2)
```




    array([[ 1.5,  4. ],
           [ 9. ,  4. ],
           [10. , 18. ]])



## Adding / removing elements 


```python
h.resize((2,6))   # not proper work error 
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    c:\Desktop\vandan\code\py\1Numpy\numpy.ipynb Cell 106 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Desktop/vandan/code/py/1Numpy/numpy.ipynb#Y214sZmlsZQ%3D%3D?line=0'>1</a> h.resize((2,6))   # not proper work error 
    

    ValueError: cannot resize an array that references or is referenced
    by another array in this way.
    Use the np.resize function or refcheck=False



```python
np.append(h,g)
```




    array([ 1. ,  2. ,  3. ,  1.5,  4. ,  9. ,  4. , 10. , 18. ])




```python
np.insert(a,1,5)
```




    array([1, 5, 2, 3])




```python
np.delete(a,[1])
```




    array([1, 3])



## Combining Array


```python
np.concatenate((a,d),axis=0)
```




    array([ 1,  2,  3, 18, 23])




```python
np.vstack([1,2,3,10,15,20])
```




    array([[ 1],
           [ 2],
           [ 3],
           [10],
           [15],
           [20]])




```python
np.r_[e,f]
```




    array([[7., 7.],
           [7., 7.],
           [1., 0.],
           [0., 1.]])




```python
np.hstack((e,f))
```




    array([[7., 7., 1., 0.],
           [7., 7., 0., 1.]])




```python
np.column_stack((b,b))
```




    array([[1.5, 2. , 3. , 1.5, 2. , 3. ],
           [4. , 5. , 6. , 4. , 5. , 6. ]])




```python
np.c_[b,b]
```




    array([[1.5, 2. , 3. , 1.5, 2. , 3. ],
           [4. , 5. , 6. , 4. , 5. , 6. ]])



## splitting Array


```python
np.hsplit(a,3)
```




    [array([1]), array([2]), array([3])]




```python
np.vsplit(c,2)
```




    [array([[[1.5, 2. , 1. ],
             [4. , 5. , 6. ]]]),
     array([[[3., 2., 3.],
             [4., 5., 6.]]])]




```python

```
