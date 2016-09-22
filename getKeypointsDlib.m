function [Keypoints] = getKeypointsDlib(files)
    NLANDMARKS = 68;
    dpath = 'dlib-18.18/examples';
    command = sprintf('%s/faceVideoLandmarks %s/faces/shape_predictor_68_face_landmarks.dat landmarks.txt', dpath, dpath);
    filesWSpaces = files;
    for ii = 1:length(filesWSpaces)
        filesWSpaces{ii} = sprintf('%s ', filesWSpaces{ii});
    end
    command = [command, ' ', cell2mat(filesWSpaces)];
    system(command);
    fin = fopen('landmarks.txt');
    X = cell2mat(textscan(fin, '(%f, %f)'));
    Keypoints = cell(1, length(files));
    for ii = 1:length(files)
        Keypoints{ii} = X((1:NLANDMARKS) + (ii-1)*NLANDMARKS, :);
    end
    fclose(fin);
end