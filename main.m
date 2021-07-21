%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Detection and Classification of Speed Limit Signs
% Author : Saikrishna Javvadi
% References : EE551 Weeks 6-8 Lecture notes
%              Examples and code from “Fundamentals of Image Processing: A Practical Approach with Example in Matlab”
%              https://in.mathworks.com/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars; close all; clc;  

% the path of the directory that contains the images to be tested
Images_Dir = '50Kph';
gold_digits_dir = 'images\GoldDigits';

% Calling the function to classify the speed limit signs
classifier(Images_Dir,gold_digits_dir)


% function to classify the images to their speed limit sign
function classifier(Images_Dir,gold_digits_dir)
    
    % Extracting number from the string to compare for the ground truth
    true_value = str2num( regexprep( Images_Dir, {'\D*([\d\.]+\d)[^\d]*', '[^\d\.]*'}, {'$1 ', ' '} ) );  
   
    % loading the images from the directory
    targetFolder = append('images/', Images_Dir);
    images = GetFileData_FromDirectory(targetFolder, '*.jpg');
    % total number of images in the directory
    total_images = length(images);
    % defining a prediction array to store predicted values 
    prediction_array = zeros(1,total_images);      

    %Iterating through all images in the directory
    for i = 1:total_images
        % Get file name and read the image
        file_name = fullfile(targetFolder, images(i).name);
        image = imread(file_name);
        
        % pre-processing the image
        pre_processed_image = pre_process(image);
        
        % Extracting speed sign from the image 
        sign_image = sign_extraction(pre_processed_image);  
            
        % Extracting the number from the image
        number_image = number_extraction(sign_image);   
        
        % comparing the sign to true values and finding the best match in
        % the gold dataset of digits
        match = best_match(number_image , gold_digits_dir);
        
        if match == true_value
            prediction_array(i) = 1;
        end

        %printing the predicted and actual labels
        fprintf('%d: Predicted_Label: %d | True_Label: %d \n', i, match, true_value);

    end

    % Calculating accuracy for the taken Speed sign
    acc = (sum(prediction_array)/total_images)*100;
    fprintf('Accuracy for %dkph Speed Sign is : %.2f%%\n', true_value,acc);

end

% function to pre-process the image to make it better for analysis or
% performing our operations in further steps
function processed_rgb_image = pre_process(image)

    % Using histogram equalisation to improve image contrast
    image = histeq(image);
    
    % converting the image to greyscale
    grey_image = rgb2gray(image);
    
    % Adjusting the image to boost brightness if the 
    % average image pixel or brightness is below a specific threshold.
    if mean(grey_image(:)) < 105
     grey_image = imadjust(grey_image);
    end
   
    % coverting rgb image back to grey scale
    processed_rgb_image = cat(3, grey_image, grey_image, grey_image);
    
end


% function to extract Speed sign from an image
% firstly the red pixels are seperated within the image and then use them 
%to crop the sign image
function interested_region = sign_extraction(rgb_image)  
    
    % resizing all the images to the same size
    image = imresize(rgb_image,[600 600]);
    
    % Converting the RGB image to HSV
    HSV_image = rgb2hsv(image);
    
    % Converting the image to HSV colorspace and 
    % thresholding the hue and saturation channels to highlight red red pixels
    red_BW = ( HSV_image(:, :, 1) <= 0.15 | HSV_image(:, :, 1) >= 0.85 ) &...
        ( HSV_image(:, :, 2) <= 1.0 & HSV_image(:, :, 2) >= 0.5);
    
    % removing imperfections in the image eliminating connected components with less than 20 pixels in area
    red_BW = bwareaopen(red_BW, 20);

    % getting the properties for connected components
    props = regionprops('table', red_BW, 'Area','Centroid', 'EquivDiameter'); 

    % removing regions which ahve area less than 150 
    props(props.Area < 150,:) = [];

    % getting the radii from the properties
    rad = props.EquivDiameter/2;

    props = sortrows(props, 1, 'descend');

    % Obtaining statistics for the largest region
    if (height(props) == 1)
        final_radius = rad;
        cent = props.Centroid(1,:);
    elseif (height(props) > 1)
        final_radius = max(rad);
        cent = props.Centroid(1,:);
    else
        final_radius = 150;
        cent = [240,240];
    end

    % obtaining the co-ordinates of centroid
    x_cent = cent(:,1);
    y_cent = cent(:,2);

    % cropping the image to get the bounding box for the region of interest
    interested_region = imcrop(image, [x_cent - final_radius, y_cent- final_radius,...
        (x_cent + final_radius)-(x_cent - final_radius),...
        (y_cent + final_radius)-(y_cent- final_radius)]);
    
end


% function to extract digits from the image and then crop it to the most
% relevant digit for the given speed sign
function interested_region = number_extraction(image)  

    % converting the image to greyscale
    grey_image = rgb2gray(image); 
    
    % increasing contrast of the image with histogram equalisation
    grey_image = histeq(grey_image);
    
    % Converting to a binary image
    mask = imbinarize(grey_image,0.3);
    % inverting the image
    mask = ~mask;
   
    % getting the properties for connected components
    props = regionprops('table', mask, 'Area', 'Centroid', 'BoundingBox'); 
    
    % deleting too small and large components
    small_comp = props.Area < 2500;
    large_comp = props.Area > 13000;
    del = logical(small_comp + large_comp);
    props(del,:) = [];
    
    % Reference coordinate to compare each proposed region
    ref_centroid = [70 140]; 
    
    if isempty(props) || height(props)==0
        % in-case there's no number found in the interested region
        interested_region = mask;
    else
        % closest distance between reference centroid and region centroid
        closest_distance = 600; 
        % initialising bounding box array for the number/digit
        bounding_box = [];       

        % Iterating through every bounding box and 
        % looking for the closest centroid w.r.to reference centroid
        for i = 1:height(props)
            distance = norm(ref_centroid-props.Centroid(i));
            if distance < closest_distance
                closest_distance = distance;
                bounding_box = props.BoundingBox(i,:);
            end
        end

        % getting the bounding box coordinate values
        x_1 = bounding_box(1,1); 
        y_1 = bounding_box(1,2);
        x_2 = x_1 + bounding_box(1,3); 
        y_2 = y_1 + bounding_box(1,4);

        % cropping the region of interest i.e the number/digit from the image
        interested_region = imcrop(mask,[x_1, y_1, x_2-x_1, y_2-y_1]); 
 
    end

end


% Function to load images from a given directory
function result = GetFileData_FromDirectory(Path,fileExtension)
% checking if the folder actually exists
if ~isfolder(Path)
    disp('no such directory exists');
    return;
end
% obtaining all image files with the given extension from the directory
file_path = fullfile(Path, fileExtension);
result = dir(file_path);
end


% function to obtain the speed limit sign by getting its best match from
% the list of available gold standard digit images
function result = best_match(normal_image, gold_digits_dir)
     
     normal_image = imresize(normal_image, [200 200]); % Images must be same size

    % Loading the gold numbers/digits extracted from the gold standard images
    gold_num_file_data = GetFileData_FromDirectory(gold_digits_dir, '*.png');
    
    black_pixels_percentage = zeros(1,length(gold_num_file_data)); 

    % Iterating over all the gold standard digits to find the best match
    for i = 1:length(gold_num_file_data)
        
        % obtaining the file path
        filePath = fullfile(gold_digits_dir, gold_num_file_data(i).name);
        %reading the image
        gold_num_image = imread(filePath);   
        
        gold_num_image = imresize(gold_num_image, [200 200]);

        % Subtracting the test image from gold standard image
        output_image = imsubtract(gold_num_image, normal_image);
        
        % Calculate the percentage of black pixels in the binary image
        black_pixels_percentage(i) = 100*(sum(output_image(:)==0)/numel(output_image(:)));
    end
    
    [~, index] = max(black_pixels_percentage);
    
    % a list of all possible values for the speed
    values_list = {100 20 30 50 80}; 
    
    % return the best match among the available ones from the list
    result = values_list{index};
 
end