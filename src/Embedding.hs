{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}


module Embedding
  ( Config(..)
  , produceEmbeddings
  , readEmbeddings
  , parseEmbeddings
  ) where


import           GHC.TypeNats (KnownNat)

import           Control.Monad (forM_)

import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy as L
import qualified Data.Text.Lazy.IO as L

import qualified Numeric.LinearAlgebra.Static as LA

import qualified Format.Cupt as Cupt
import qualified Embedding.Dict as D
import           Net.Basic (toList)


-----------------------------------
-- Creation
-----------------------------------


data Config = Config
  { embPath :: FilePath
    -- ^ Embedding file
  , inpPath :: FilePath
    -- ^ Input .cupt file
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


-----------------------------------
-- Loading
-----------------------------------


-- | Read the embeddings file.
readEmbeddings
  :: (KnownNat d)
  => FilePath -- ^ Embedding file, one vector per .cupt word
  -> IO [Maybe [LA.R d]]
readEmbeddings = fmap parseEmbeddings . L.readFile


-- | Parse the embeddings.  Returns `Nothing` if one of the words in a
-- sentence has no corresponding vector embedding.
parseEmbeddings
  :: (KnownNat d)
  => L.Text -- ^ Embedding file, one vector per .cupt word
  -> [Maybe [LA.R d]]
parseEmbeddings
  = map parseSent
  . filter (not . L.null)
  . L.splitOn "\n\n"


parseSent :: (KnownNat d) => L.Text -> Maybe [LA.R d]
parseSent
  = sequence
  . map parseToken
  . L.lines


parseToken :: (KnownNat d) => L.Text -> Maybe (LA.R d)
parseToken line =
  let w : vs0 = L.words line
      vs = map (read . L.unpack) vs0
   in case vs of
        [] -> Nothing
        _  -> Just (LA.fromList vs)
