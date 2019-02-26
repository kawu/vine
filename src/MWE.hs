{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}


module MWE 
  ( Sent(..)

  -- * Training
  , train

  -- * Tagging
  , tag
  , tagMany
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)

import           Control.Monad (guard, when, forM_)

import qualified System.Directory as D
import           System.FilePath ((</>))

import           Numeric.LinearAlgebra.Static.Backprop (R)
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.Backprop as BP

import           Dhall -- (Interpret)
import qualified Data.Aeson as JSON

import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as B
-- import           Codec.Compression.Zlib (compress, decompress)
import qualified Data.IORef as R

import qualified Format.Cupt as Cupt
import qualified Net.ArcGraph as Net
import           Net.ArcGraph (Elem)
-- import qualified Net.MWE2 as MWE
import qualified Embedding as Emb
-- import qualified GradientDescent.Momentum as Mom
import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.AdaDelta as Ada
-- import qualified SGD as SGD

-- import Debug.Trace (trace)


----------------------------------------------
-- Data
----------------------------------------------


-- | Input sentence
data Sent d = Sent
  { cuptSent :: Cupt.Sent
    -- ^ The .cupt sentence
  , wordEmbs :: [R d]
    -- ^ The corresponding word embeddings
  } deriving (Show, Generic, Binary)



-- | Convert the given Cupt file to a training dataset element.
--   * `d` -- embedding size (dimension)
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> (Elem d 2, M.Map Cupt.TokID G.Vertex)
--   -> (Elem d, M.Map Cupt.TokID G.Vertex)
mkElem mweSel Sent{..} =

  (createElem nodes arcs, idMap)

  where

    -- Helpers
    idMap = M.fromList
      [ (Cupt.tokID tok, i)
      | tok <- cuptSent
      , let (Cupt.TokID i) = Cupt.tokID tok
      ]
    tokID tok = idMap M.! Cupt.tokID tok
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSent 
      ]
    tokPar tok = tokMap M.! Cupt.dephead tok
    isRoot tok = Cupt.dephead tok == Cupt.TokID 0
    getMweIDs tok
      = S.fromList
      . map fst
      . filter (mweSel . snd)
      $ Cupt.mwe tok

    -- Graph nodes and labeled arcs
    nodes = do
      (tok, vec) <- zip cuptSent wordEmbs
      return (tokID tok, vec)
    arcs = do
      tok <- cuptSent
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
          -- The arc label is `True` if the token and its parent are annoted as
          -- a part of the same MWE of the appropriate MWE category
          arcLabel =
            if (not . S.null)
                 ((getMweIDs tok) `S.intersection` (getMweIDs par))
               then True
               else False
      return ((tokID tok, tokID par), arcLabel)


-- | Create a dataset element based on nodes and labeled arcs.
createElem
  :: (KnownNat d)
  => [(G.Vertex, R d)]
  -> [(Net.Arc, Bool)]
  -> Elem d 2
--   -> Elem d
createElem nodes arcs = Net.Elem
  { graph = graph
  , labMap = valMap
  }
  where
    vertices = [v | (v, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      (map fst arcs)
    lbMap = M.fromList nodes
    graph = Net.Graph
      { Net.graphStr = gStr
      , Net.graphInv = G.transposeG gStr
      , Net.labelMap = lbMap }
    valMap = M.fromList $ do
      (arc, isMwe) <- arcs
      return 
        ( arc
--         , if isMwe then 1.0 else 0.0
        , if isMwe
             then mwe
             else notMwe
        )


-- | Is MWE or not?
mwe, notMwe :: R 2
notMwe = LA.vector [1, 0]
mwe = LA.vector [0, 1]


----------------------------------------------
-- Training
----------------------------------------------


instance JSON.FromJSON SGD.Method
instance JSON.ToJSON SGD.Method where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret SGD.Method

instance JSON.FromJSON SGD.Config
instance JSON.ToJSON SGD.Config where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret SGD.Config


-- | Train the MWE identification network.
train
  :: SGD.Config
    -- ^ General training confiration
  -> Int
    -- ^ Graph net recursion depth
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent 300]
    -- ^ Training dataset
  -> Net.Param 300 2
--   -> Net.Param 300
    -- ^ Initial network
  -> IO (Net.Param 300 2)
--   -> IO (Net.Param 300)
train sgdCfg depth mweTyp cupt net0 = do
  -- dataSet <- mkDataSet (== mweTyp) tmpDir cupt
  let cupt' = map (fst . mkElem (== mweTyp)) cupt
  SGD.withDisk cupt' $ \dataSet -> do
    putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
    -- net0 <- Net.new 300 2
    -- trainProgSGD sgdCfg dataSet globalDepth net0
    SGD.sgd sgdCfg dataSet gradient quality net0
  where
    gradient xs = BP.gradBP (Net.netError xs $ fromIntegral depth)
    quality x = BP.evalBP (Net.netError [x] $ fromIntegral depth)


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tag many sentences
tagMany
  :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Net.Param 300 2  -- ^ Network parameters
--   -> Net.Param 300    -- ^ Network parameters
  -> Int              -- ^ Depth (see also `trainDepth`)
  -> [Sent 300]       -- ^ Cupt sentences
  -> IO ()
tagMany mweTyp net depth cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tag mweTyp net depth sent


-- | Tag (output the result on stdin).
tag
  :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Net.Param 300 2  -- ^ Network parameters
--   -> Net.Param 300    -- ^ Network parameters
  -> Int              -- ^ Depth (see also `trainDepth`)
  -> Sent 300         -- ^ Cupt sentence
  -> IO ()
tag mweTyp net depth sent = do

  forM_ (M.toList arcMap) $ \(e, v) -> do
    when (isMWE v) (printArc e)

  where

    (elem, tokMap) = mkElem (const False) sent
    arcMap = Net.eval net depth (Net.graph elem)
    isMWE statVect =
      let vect = LA.unwrap statVect
          val = vect `LAD.atIndex` 1
       in val > 0.5
--     isMWE = (> 0.5)

    wordMap = M.fromList $ do
      tok <- cuptSent sent
      return (tokMap M.! Cupt.tokID tok, Cupt.orth tok)

    printArc (p, q) = do
      T.putStr (wordMap M.! p)
      T.putStr " => "
      T.putStrLn (wordMap M.! q)
