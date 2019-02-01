import pytraph as pt
 
 
string  = pt.constant("Hello World")
 
 
with pt.Session() as sess:
    print(sess.run(string))