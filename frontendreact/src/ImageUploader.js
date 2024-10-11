import React, { useState } from 'react';
import axios from 'axios';

function ImageUploader() {
  const [file, setFile] = useState(null);
  const [equation, setEquation] = useState("");
  const [solution, setSolution] = useState("");
  const [error, setError] = useState("");
  const [imageURL, setImageURL] = useState(null);

  const onFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setImageURL(URL.createObjectURL(selectedFile)); // Créer un URL pour afficher l'image
  };

  const onUpload = async () => {
    if (!file) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setEquation(response.data.equation)
      setSolution(response.data.solution)
      if(response.data.error)setError(response.data.error)
      
      
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className='p-4 max-w-lg mx-auto'>
      <div className='mb-4'>
        <input type='file' onChange={onFileChange}  className='block w-full text-sm text-gray-500 border border-gray-300 rounded-lg cursor-pointer focus:outline-none'/>
      </div>
      <button onClick={onUpload} className='w-full py-2 px-4 bg-blue-500 text-white font-semibold rounded-lg shadow-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400'>Upload and Predict</button>
      {imageURL && (
        <div className='flex flex-col items-center justify-center mt-4'>
          <p className='text-lg font-medium mb-2'>Selected Image:</p>
          <img
            src={imageURL}
            alt='Uploaded'
            className='max-w-full h-auto border border-gray-300 rounded-lg shadow-md'
          />
        </div>
      )}
      {equation.length > 0 && (
        <div className='flex flex-col items-center justify-center mt-4'>
          <div className='text-center'>
            <div className='text-lg font-medium mb-2'>Predicted digits:</div>
            <div className='text-xl font-semibold mb-2'>Image détécter : {equation}</div>
            <div className='text-lg font-semibold'>{error.length > 0 ? `${error}` : `x=${solution}`}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
