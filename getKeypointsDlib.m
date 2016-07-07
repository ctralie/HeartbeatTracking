function [X] = getKeypointsDlib(filename)
    dpath = 'dlib-18.18/examples';
    command = sprintf('%s/build/jh_face_landmark_ex %s/faces/shape_predictor_68_face_landmarks.dat %s temp.txt', dpath, dpath, filename);
    system(command);
    fin = fopen('temp.txt');
    X = cell2mat(textscan(fin, '(%f, %f)'));
    fclose(fin);
end