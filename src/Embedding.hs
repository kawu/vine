{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}


module Embedding
  ( Config(..)
  , produceEmbeddings
  ) where


import           GHC.TypeNats (KnownNat)

import           Control.Monad (forM_)

import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as T

import qualified Numeric.LinearAlgebra.Static as LA

import qualified Format.Cupt as Cupt
import qualified Embedding.Dict as D
import           Net.Basic (toList)


data Config = Config
  { embPath :: FilePath
    -- ^ Embedding file
  , inpPath :: FilePath
    -- ^ Input .cupt file
  , outPath :: FilePath
    -- ^ Output file (TODO: remove and use stdin?)
  , embNoHeader :: Bool
    -- ^ Does the embedding file has header?
  } deriving (Show, Eq, Ord)


-- | Take a file with embeddings, a .cupt file, and produce the corresponding
-- file with the determined embeddings for the individual words.
produceEmbeddings
  :: Config
  -> IO ()
produceEmbeddings Config{..} = do
  vocab <- determineVocab . concat . concat <$> Cupt.readCupt inpPath
  -- putStr "VOCAB:" >> print vocab
  -- TODO: make `300` parametric
  dict <- D.loadSel (`S.member` vocab) (not embNoHeader) embPath :: IO (D.Dict 300)
  cupt <- Cupt.readCupt inpPath
  forM_ cupt $ \par -> do
    forM_ par $ \sent -> do
      forM_ sent $ \tok -> do
        case M.lookup (Cupt.orth tok) dict of
          Nothing -> putStrLn "#"
          Just v -> do
            T.putStr (Cupt.orth tok) >> T.putStr "\t"
            T.putStrLn . T.intercalate "\t" . map (T.pack . show) $ toList v
    putStrLn ""


-- | Determine the vocabulary in the .cupt file.
determineVocab :: [Cupt.GenToken mwe] -> S.Set T.Text
determineVocab = S.fromList . map Cupt.orth
