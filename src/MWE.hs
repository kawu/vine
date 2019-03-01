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
  , depRelsIn
  , posTagsIn

  -- * Tagging
  , tag
  , tagMany
  ) where


import           Prelude hiding (elem)

import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)

import           Control.Monad (guard, forM_)

-- import qualified System.Directory as D
-- import           System.FilePath ((</>))

import           Numeric.LinearAlgebra.Static.Backprop (R)
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.Backprop as BP

import           Dhall -- (Interpret)
import qualified Data.Aeson as JSON

import qualified Data.Foldable as Fold
import           Data.Semigroup (Max(..))
import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as L
import           Data.Binary (Binary)
-- import qualified Data.Binary as Bin
-- import qualified Data.ByteString.Lazy as B
-- import           Codec.Compression.Zlib (compress, decompress)
-- import qualified Data.IORef as R

import qualified Format.Cupt as Cupt
import qualified Net.ArcGraph as Net
import           Net.ArcGraph (Elem)
-- import qualified Net.MWE2 as MWE
-- import qualified Embedding as Emb
-- import qualified GradientDescent.Momentum as Mom
import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Momentum as Mom
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



-- | Convert the given Cupt file to a training dataset element.  The token IDs
-- are used as vertex identifiers in the resulting graph.
--
--   * `d` -- embedding size (dimension)
--
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> Elem d 2 DepRel POS
mkElem mweSel Sent{..} =

  createElem nodes arcs

  where

    -- A map from token IDs to tokens
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSent 
      ]
    -- The parent of the given token
    tokPar tok = tokMap M.! Cupt.dephead tok
    -- Is the token the root?
    isRoot tok = Cupt.dephead tok == Cupt.TokID 0
    -- The IDs of the MWEs present in the given token
    getMweIDs tok
      = S.fromList
      . map fst
      . filter (mweSel . snd)
      $ Cupt.mwe tok

    -- Graph nodes: a list of token IDs and the corresponding vector embeddings
    nodes = do
      (tok, vec) <- zip cuptSent wordEmbs
      let node = Net.Node
            { nodeEmb = vec
            , nodeLab = Cupt.upos tok
            -- | TODO: could be lemma?
            , nodeLex = Cupt.orth tok
            }
      return (tokID tok, node)
    -- Labeled arcs of the graph
    arcs = do
      tok <- cuptSent
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
          -- The arc value is `True` if the token and its parent are annoted as
          -- a part of the same MWE of the appropriate MWE category
          arcVal =
            if (not . S.null)
                 ((getMweIDs tok) `S.intersection` (getMweIDs par))
               then True
               else False
      return ((tokID tok, tokID par), Cupt.deprel tok, arcVal)


-- | Create a dataset element based on nodes and labeled arcs.
createElem
  :: (KnownNat d)
  -- => [(G.Vertex, R d, POS)]
  => [(G.Vertex, Net.Node d POS)]
  -> [(Net.Arc, DepRel, Bool)]
  -> Elem d 2 DepRel POS
createElem nodes arcs = Net.Elem
  { graph = graph
  , labMap = valMap
  }
  where
    vertices = [v | (v, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      (map _1 arcs)
    graph = Net.Graph
      { Net.graphStr = gStr
      , Net.graphInv = G.transposeG gStr
      , Net.nodeLabelMap = M.fromList $ nodes
          -- map (\(x, e, pos) -> (x, Net.Node e pos)) nodes
      , Net.arcLabelMap = M.fromList $
          map (\(x, y, _) -> (x, y)) arcs
      }
    valMap = M.fromList $ do
      (arc, _arcLab, isMwe) <- arcs
      return 
        ( arc
--         , if isMwe then 1.0 else 0.0
        , if isMwe
             then mwe
             else notMwe
        )
    _1 (x, _, _) = x
    -- _2 (_, y, _) = y
    -- _3 (_, _, z) = z


-- | Is MWE or not?
mwe, notMwe :: R 2
notMwe = LA.vector [1, 0]
mwe = LA.vector [0, 1]


-- | Token ID
tokID :: Cupt.Token -> Int
tokID tok =
  case Cupt.tokID tok of
    Cupt.TokID i -> i
    Cupt.TokIDRange _ _ ->
      error "MWE.tokID: token ID ranges not supported"


----------------------------------------------
-- Training
----------------------------------------------


instance JSON.FromJSON Mom.Config
instance JSON.ToJSON Mom.Config where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret Mom.Config

instance JSON.FromJSON Ada.Config
instance JSON.ToJSON Ada.Config where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret Ada.Config

instance JSON.FromJSON SGD.Method
instance JSON.ToJSON SGD.Method where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret SGD.Method

instance JSON.FromJSON SGD.Config
instance JSON.ToJSON SGD.Config where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret SGD.Config


-- | Depdency relation
type DepRel = T.Text


-- | POS tag
type POS = T.Text


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
  -> Net.Param 300 2 DepRel POS
--   -> Net.Param 300
    -- ^ Initial network
  -> IO (Net.Param 300 2 DepRel POS)
--   -> IO (Net.Param 300)
train sgdCfg depth mweTyp cupt net0 = do
  -- dataSet <- mkDataSet (== mweTyp) tmpDir cupt
  let cupt' = map (mkElem (== mweTyp)) cupt
  SGD.withDisk cupt' $ \dataSet -> do
    putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
    -- net0 <- Net.new 300 2
    -- trainProgSGD sgdCfg dataSet globalDepth net0
    SGD.sgd sgdCfg dataSet gradient quality net0
  where
    gradient x = BP.gradBP (Net.netError [x] $ fromIntegral depth)
    quality x = BP.evalBP (Net.netError [x] $ fromIntegral depth)


-- | Extract dependeny relations present in the given dataset.
depRelsIn :: [Cupt.GenSent mwe] -> S.Set DepRel
depRelsIn =
  S.fromList . concatMap extract 
  where
    extract = map Cupt.deprel


-- | Extract dependeny relations present in the given dataset.
posTagsIn :: [Cupt.GenSent mwe] -> S.Set POS
posTagsIn =
  S.fromList . concatMap extract 
  where
    extract = map Cupt.upos


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tag many sentences
tagMany
  :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Net.Param 300 2 DepRel POS
                      -- ^ Network parameters
  -> Int              -- ^ Depth (see also `trainDepth`)
  -> [Sent 300]       -- ^ Cupt sentences
  -> IO ()
tagMany mweTyp net depth cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tag mweTyp net depth sent


-- -- | Tag (output the result on stdin).
-- tag
--   :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
--   -> Net.Param 300 2  -- ^ Network parameters
-- --   -> Net.Param 300    -- ^ Network parameters
--   -> Int              -- ^ Depth (see also `trainDepth`)
--   -> Sent 300         -- ^ Cupt sentence
--   -> IO ()
-- tag mweTyp net depth sent = do
-- 
--   forM_ (M.toList arcMap) $ \(e, v) -> do
--     when (isMWE v) (printArc e)
-- 
--   where
-- 
--     elem = mkElem (const False) sent
--     arcMap = Net.eval net depth (Net.graph elem)
--     isMWE statVect =
--       let vect = LA.unwrap statVect
--           val = vect `LAD.atIndex` 1
--        in val > 0.5
-- --     isMWE = (> 0.5)
-- 
--     wordMap = M.fromList $ do
--       tok <- cuptSent sent
--       return (tokID tok, Cupt.orth tok)
-- 
--     printArc (p, q) = do
--       T.putStr (wordMap M.! p)
--       T.putStr " => "
--       T.putStrLn (wordMap M.! q)


-- | Tag (output the result on stdin).
tag
  :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Net.Param 300 2 DepRel POS
                      -- ^ Network parameters
  -> Int              -- ^ Depth (see also `trainDepth`)
  -> Sent 300         -- ^ Cupt sentence
  -> IO ()
tag mweTyp net depth sent = do
  L.putStrLn $ Cupt.renderPar [Cupt.abstract sent']
  where
    elem = mkElem (const False) sent
    arcMap = Net.eval net depth (Net.graph elem)
    sent' = annotate mweTyp (cuptSent sent) arcMap


----------------------------------------------
-- Annotation
----------------------------------------------


-- | Annotate the sentence with the given MWE type, given the network
-- evaluation results.
annotate
  :: Cupt.MweTyp
    -- ^ MWE category
  -> Cupt.Sent
    -- ^ Input .cupt sentence
  -> M.Map Net.Arc (R 2)
    -- ^ Net evaluation results
  -> Cupt.Sent
annotate mweTyp cupt arcMap =

  map enrich cupt

  where

    -- Enrich the token with new MWE information
    enrich tok =
      case M.lookup (tokID tok) mweIdMap of
        Nothing -> tok
        Just mweId ->
          let newMwe = (mweId, mweTyp)
           in tok {Cupt.mwe = newMwe : Cupt.mwe tok}

    -- Determine the set of MWE arcs
    arcSet = S.fromList $ do
      (arc, v) <- M.toList arcMap
      guard $ isMWE v
      return arc

    isMWE statVect =
      let vect = LA.unwrap statVect
          val = vect `LAD.atIndex` 1
       in val > 0.5

    -- Determine the mapping from nodes to new MWE id's
    ccs = findConnectedComponents arcSet
    mweIdMap = M.fromList . concat $ do
      (cc, mweId) <- zip ccs [maxMweID cupt + 1 ..]
      (v, w) <- S.toList cc
      return [(v, mweId), (w, mweId)]


-- | Given a set of graph arcs, determine all the connected arc subsets in the
-- corresponding graph.
findConnectedComponents
  :: S.Set Net.Arc
  -> [S.Set Net.Arc]
findConnectedComponents arcSet
  | S.null arcSet = []
  | otherwise
      -- Some components can be empty!  Perhaps because the graph is sparse?
      = filter (not . S.null)
      $ map (arcsInTree arcSet) (G.components graph)
  where
    vertices = S.toList (nodesIn arcSet)
    graph = G.buildG
      (minimum vertices, maximum vertices)
      (S.toList arcSet)


-- | Determine the set of arcs in the given connected graph component.
-- TODO: This could be done much more efficiently!
arcsInTree
  :: S.Set Net.Arc
    -- ^ The set of all arcs
  -> G.Tree G.Vertex
    -- ^ Connected component
  -> S.Set Net.Arc
arcsInTree arcSet cc = S.fromList $ do
  (v, w) <- S.toList arcSet
  guard $ v `S.member` vset
  guard $ w `S.member` vset
  return (v, w)
  where
    vset = S.fromList (Fold.toList cc)


-- | The set of nodes in the given arc set.
nodesIn :: S.Set Net.Arc -> S.Set G.Vertex
nodesIn arcSet =
  (S.fromList . concat)
    [[v, w] | (v, w) <- S.toList arcSet]


-- | Determine the maximum mwe ID present in the given sentence.
maxMweID :: Cupt.Sent -> Int
maxMweID =
  getMax . Fold.foldMap mweID
  where
    mweID tok = case Cupt.mwe tok of
      [] -> Max 0
      xs -> Max . maximum $ map fst xs


-- report x = trace (show x) x
