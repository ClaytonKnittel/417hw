
d = [1 2; 2 3; 0 1; 4 4];
l = ['a'; 'b'; 'b'; 'b'];

tree = fitctree(d,l,'CrossVal','off');

tree.predict([1 2; 2 3; 0 0; 10 10])