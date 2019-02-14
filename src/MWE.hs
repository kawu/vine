{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}


module MWE 
  ( train
  , tag
  , tagMany
  ) where


import           GHC.TypeNats (KnownNat)

import           Control.Monad (guard, when, forM_)

import           Numeric.LinearAlgebra.Static.Backprop (R)
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.Backprop as BP

import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T

import qualified Format.Cupt as Cupt
import qualified Net.ArcGraph as Net
import qualified Net.MWE2 as MWE
import qualified Embedding as Emb
import qualified GradientDescent.Momentum as Mom

-- import Debug.Trace (trace)


----------------------------------------------
-- Dataset
----------------------------------------------


-- | Dataset element
--
-- TODO: add mapping to original token IDs
--
data Elem d = Elem
  { graph :: Net.Graph (R d)
    -- ^ Input graph
  , labMap :: M.Map Net.Arc (R 2)
    -- ^ Target labels
  , tokMap :: M.Map Cupt.TokID G.Vertex
    -- ^ Token ID -> graph vertices mapping
  } deriving (Show)


-- | Convert the given Cupt file to a training dataset element.
--   * `d` -- embedding size (dimension)
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Cupt.Sent
    -- ^ Input sentence
  -> [R d]
    -- ^ The corresponding list of embeddings
  -> Elem d
mkElem mweSel sent embs =

  createElem idMap nodes arcs

  where

    -- Helpers
    idMap = M.fromList
      [ (Cupt.tokID tok, i)
      | tok <- sent
      , let (Cupt.TokID i) = Cupt.tokID tok
      ]
    tokID tok = idMap M.! Cupt.tokID tok
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- sent 
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
      (tok, vec) <- zip sent embs
      return (tokID tok, vec)
    arcs = do
      tok <- sent
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
  => M.Map Cupt.TokID G.Vertex
  -> [(G.Vertex, R d)]
  -> [(Net.Arc, Bool)]
  -> Elem d
createElem idMap nodes arcs = Elem
  { tokMap = idMap
  , graph = graph
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
        , if isMwe
             then MWE.mwe
             else MWE.notMwe
        )


----------------------------------------------
-- Training
----------------------------------------------


-- data TrainConfig = TrainConfig
--   { 
--   } deriving (Show, Eq, Ord)


-- | Train the MWE identification network.
train
  :: (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> [Cupt.Sent]   -- ^ Cupt
  -> [[LA.R 300]]  -- ^ Embeddings
  -> IO (Net.Param 300 2)
train mweSel cupt embss = do
  net0 <- Net.new 300 2
  Net.trainProg (gdCfg trainData) 1 net0
  where
    trainData = do
      (sent, embs) <- zip cupt embss
      let Elem{..} = mkElem mweSel sent embs
      return (graph, labMap)
    -- Gradient descent configuration
    gdCfg dataSet depth = Mom.Config
      { iterNum = 100
      , gradient = BP.gradBP (Net.netError dataSet depth)
      , quality = BP.evalBP (Net.netError dataSet depth)
      , reportEvery = 1
      , gain0 = 0.05 / fromIntegral (depth+1)
      , tau = 50
      , gamma = 0.0 ** fromIntegral (depth+1)
      }


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tag many sentences
tagMany
  :: Net.Param 300 2  -- ^ Network parameters
  -> Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> [Cupt.Sent]      -- ^ Cupt sentence
  -> [[LA.R 300]]     -- ^ Embeddings
  -> IO ()
tagMany net mweTyp cupt embss = do
  forM_ (zip cupt embss) $ \(sent, embs) -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth sent
    tag net mweTyp sent embs


-- | Tag (output the result on stdin).
tag
  :: Net.Param 300 2  -- ^ Network parameters
  -> Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Cupt.Sent        -- ^ Cupt sentence
  -> [LA.R 300]       -- ^ Embeddings
  -> IO ()
tag net mweTyp sent embs = do

  forM_ (M.toList arcMap) $ \(e, v) -> do
    when (isMWE v) (printArc e)

  where

    elem = mkElem (const False) sent embs
    arcMap = Net.eval net 1 (graph elem)
    isMWE statVect =
      let vect = LA.unwrap statVect
          val = vect `LAD.atIndex` 1
       in val > 0.5

    wordMap = M.fromList $ do
      tok <- sent
      return (tokMap elem M.! Cupt.tokID tok, Cupt.orth tok)

    printArc (p, q) = do
      T.putStr (wordMap M.! p)
      T.putStr " => "
      T.putStrLn (wordMap M.! q)
